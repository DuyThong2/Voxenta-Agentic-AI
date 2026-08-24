import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from config import ai_usage_pricing as pricing
from events.question_asset_analysis_requested import QuestionAssetAnalysisRequestedEvent
from infra.storage.audio_storage import download_from_s3
from utils.speech_client import transcribe

logger = logging.getLogger(__name__)


class AssetAnalysisResult(BaseModel):
    transcript: Optional[str] = None
    description: Optional[str] = None


def _log_llm_cost(response, *, site: str) -> None:
    """Log-only chi phí gpt-4o cho asset analysis -- KHÔNG gọi ai_usage_tracker.record_llm_usage:
    chi phí này thuộc về việc soạn câu hỏi (1 lần/asset), không gắn với answer_id/phiên thi-luyện
    nào, nên không có chỗ trong ledger ai_usage_record hiện tại (khoá cứng theo examSessionId+
    turnId). Đây chỉ để nhìn thấy được chi phí trong log, không phải nguồn trừ quota.

    Đơn giản hoá: không trừ cache tokens như ai_usage_tracker.py làm cho luồng thi/luyện -- các lời
    gọi ở đây là one-off, không có prompt lặp lại đủ dài để cache thật sự phát sinh.
    """
    usage_metadata = getattr(response, "usage_metadata", None)
    if not usage_metadata:
        return
    input_tokens = usage_metadata.get("input_tokens") or 0
    output_tokens = usage_metadata.get("output_tokens") or 0
    price = pricing.llm_price_for("gpt-4o")
    cost_usd = (input_tokens / 1_000_000) * price.input_per_mtok + (output_tokens / 1_000_000) * price.output_per_mtok
    logger.info(
        "[asset_analysis] gpt-4o cost site=%s input_tokens=%d output_tokens=%d cost_usd=%.6f",
        site, input_tokens, output_tokens, cost_usd,
    )


_SYSTEM_PROMPT = """You analyze question assets for an English speaking exam authoring system.

Return only objective factual grounding that helps later evaluation stay fair.
- For images: describe visible elements only.
- For audio/video: summarize what is actually said or happens, based on transcript context.
- For text passages: describe themes, imagery, structure, or notable language in a neutral way.
- Never claim there is one uniquely correct interpretation or emotional meaning.
- Keep the description concise but useful for an examiner AI."""


def analyze_asset_request(event: QuestionAssetAnalysisRequestedEvent) -> AssetAnalysisResult:
    payload = event.payload
    asset_type = (payload.asset_type or "").upper()

    if asset_type == "IMAGE":
        return AssetAnalysisResult(
            transcript=None,
            description=_describe_image(
                asset_url=payload.url or "",
                question_text=payload.question_text,
                evaluation_guide=payload.evaluation_guide,
            ),
        )

    if asset_type == "TEXT_PASSAGE":
        transcript_text = (payload.existing_transcript or "").strip()
        if not transcript_text:
            raise ValueError("TEXT_PASSAGE analysis requires existing_transcript")
        return AssetAnalysisResult(
            transcript=None,
            description=_describe_text_passage(
                transcript_text=transcript_text,
                question_text=payload.question_text,
                evaluation_guide=payload.evaluation_guide,
            ),
        )

    if asset_type in {"VIDEO", "AUDIO"}:
        transcript_output: Optional[str] = None
        transcript_input = (payload.existing_transcript or "").strip()
        if not transcript_input:
            if not payload.url:
                raise ValueError(f"{asset_type} analysis requires asset URL when transcript is missing")
            transcript_input = _transcribe_media_asset(payload.url)
            transcript_output = transcript_input or None
        description_output = None
        if not payload.existing_description:
            description_output = _describe_transcript(
                transcript_text=transcript_input,
                question_text=payload.question_text,
                evaluation_guide=payload.evaluation_guide,
                asset_type=asset_type,
            )
        return AssetAnalysisResult(
            transcript=transcript_output,
            description=description_output,
        )

    raise ValueError(f"Unsupported asset type: {asset_type}")


def _describe_image(*, asset_url: str, question_text: Optional[str], evaluation_guide) -> str:
    llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(
        AssetAnalysisResult, include_raw=True
    )
    prompt = _build_description_prompt(
        question_text=question_text,
        evaluation_guide=evaluation_guide,
        focus="Describe only what is visibly present in the image. Do not infer a single correct meaning.",
        source_label="image",
    )
    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": asset_url}},
                {"type": "text", "text": prompt},
            ]
        ),
    ])
    _log_llm_cost(response["raw"], site="describe_image")
    result = response["parsed"]
    if result is None:
        raise ValueError(f"gpt-4o structured output parsing failed: {response.get('parsing_error')}")
    return (result.description or "").strip()


def _describe_text_passage(*, transcript_text: str, question_text: Optional[str], evaluation_guide) -> str:
    return _describe_transcript(
        transcript_text=transcript_text,
        question_text=question_text,
        evaluation_guide=evaluation_guide,
        asset_type="TEXT_PASSAGE",
    )


def _describe_transcript(
    *,
    transcript_text: str,
    question_text: Optional[str],
    evaluation_guide,
    asset_type: str,
) -> str:
    llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(
        AssetAnalysisResult, include_raw=True
    )
    focus = (
        "Summarize the media objectively from the transcript. Do not invent missing visuals or claim one correct interpretation."
        if asset_type in {"VIDEO", "AUDIO"}
        else "Describe the passage objectively: themes, imagery, structure, or notable wording. Do not claim one uniquely correct interpretation."
    )
    prompt = (
        f"{_build_description_prompt(question_text=question_text, evaluation_guide=evaluation_guide, focus=focus, source_label=asset_type.lower())}\n\n"
        f"## Source Transcript\n{transcript_text}"
    )
    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    _log_llm_cost(response["raw"], site="describe_transcript")
    result = response["parsed"]
    if result is None:
        raise ValueError(f"gpt-4o structured output parsing failed: {response.get('parsing_error')}")
    return (result.description or "").strip()


def _build_description_prompt(*, question_text: Optional[str], evaluation_guide, focus: str, source_label: str) -> str:
    lines = [
        f"You are writing an objective description for a {source_label} asset used with a speaking question.",
        focus,
    ]
    if question_text:
        lines.append(f"Question text: {question_text}")
    if evaluation_guide is not None:
        if evaluation_guide.expected_content:
            lines.append(f"Expected content guidance: {evaluation_guide.expected_content}")
        if evaluation_guide.key_points:
            lines.append(f"Key points guidance: {evaluation_guide.key_points}")
        if evaluation_guide.acceptable_responses:
            lines.append(f"Acceptable responses guidance: {evaluation_guide.acceptable_responses}")
    lines.append("Return a concise neutral description in plain English.")
    return "\n".join(lines)


def _transcribe_media_asset(asset_ref: str) -> str:
    local_path = download_from_s3(asset_ref)
    local_path_obj = Path(local_path)
    temp_dir = tempfile.TemporaryDirectory()
    try:
        wav_path = Path(temp_dir.name) / "asset-audio.wav"
        _convert_media_to_wav(local_path_obj, wav_path)
        transcript_text = transcribe(str(wav_path), "en-US")
        return (transcript_text or "").strip()
    finally:
        temp_dir.cleanup()
        if local_path != asset_ref:
            Path(local_path).unlink(missing_ok=True)


def _convert_media_to_wav(source_path: Path, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
