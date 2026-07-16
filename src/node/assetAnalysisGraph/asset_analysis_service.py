import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from events.question_asset_analysis_requested import QuestionAssetAnalysisRequestedEvent
from infra.storage.audio_storage import download_from_s3
from utils.speech_client import transcribe


class AssetAnalysisResult(BaseModel):
    transcript: Optional[str] = None
    description: Optional[str] = None


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
    llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(AssetAnalysisResult)
    prompt = _build_description_prompt(
        question_text=question_text,
        evaluation_guide=evaluation_guide,
        focus="Describe only what is visibly present in the image. Do not infer a single correct meaning.",
        source_label="image",
    )
    result = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": asset_url}},
                {"type": "text", "text": prompt},
            ]
        ),
    ])
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
    llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(AssetAnalysisResult)
    focus = (
        "Summarize the media objectively from the transcript. Do not invent missing visuals or claim one correct interpretation."
        if asset_type in {"VIDEO", "AUDIO"}
        else "Describe the passage objectively: themes, imagery, structure, or notable wording. Do not claim one uniquely correct interpretation."
    )
    prompt = (
        f"{_build_description_prompt(question_text=question_text, evaluation_guide=evaluation_guide, focus=focus, source_label=asset_type.lower())}\n\n"
        f"## Source Transcript\n{transcript_text}"
    )
    result = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
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
