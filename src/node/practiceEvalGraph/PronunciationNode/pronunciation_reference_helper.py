"""Builds a pronunciation-alignment reference text from a raw (uncorrected)
transcript -- grounded in the question/evaluation-guide/asset context so
ambiguous or garbled words are resolved toward what the question was actually
about, not a random unrelated word. Used exclusively as pronunciation_eval_node's
reference_text; every other consumer (validity, grammar/lexical/coherence eval,
UI display) keeps reading speaking_input.transcribed_text untouched."""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from infra.message_broker import ai_usage_tracker
from node.practiceEvalGraph.PronunciationNode.pronunciation_reference_prompt import SYSTEM_PROMPT
from node.state_models.speaking_input import QuestionContext
from utils.confidence_utils import (
    CLAUDE_MODEL,
    call_with_retry_and_fallback,
    compute_reference_confidence,
    llm_call_slot,
)

logger = logging.getLogger(__name__)


def _build_context_block(question: Optional[QuestionContext]) -> str:
    if question is None:
        return ""

    lines = []
    if question.question_text:
        lines.append(f'Question: "{question.question_text}"')

    guide = question.evaluation_guide
    if guide is not None:
        if guide.expected_content:
            lines.append(f"Expected content: {guide.expected_content}")
        if guide.key_points:
            lines.append(f"Key points: {guide.key_points}")
        if guide.acceptable_responses:
            lines.append(f"Acceptable responses: {guide.acceptable_responses}")
        if guide.common_mistakes:
            lines.append(f"Common mistakes: {guide.common_mistakes}")

    asset = question.asset
    if asset is not None:
        if asset.description:
            lines.append(f"Asset description: {asset.description}")
        if asset.transcript:
            lines.append(f"Asset transcript: {asset.transcript}")
        if asset.alt_text:
            lines.append(f"Asset alt text: {asset.alt_text}")

    return "\n".join(lines)


def build_pronunciation_reference(
    transcript: str,
    question: Optional[QuestionContext] = None,
    *,
    provider: str = "openai",
    answer_id: Optional[str] = None,
) -> str:
    if not transcript or not transcript.strip():
        return transcript

    model = CLAUDE_MODEL if provider == "claude" else "gpt-5.4"
    llm = (
        ChatAnthropic(model=model, temperature=0.7)
        if provider == "claude"
        else ChatOpenAI(model=model, reasoning_effort="medium")
    )

    context_block = _build_context_block(question)
    human_content = (
        f"Context:\n{context_block}\n\nTranscript:\n{transcript}"
        if context_block
        else transcript
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    with llm_call_slot():
        response = llm.invoke(messages)
    try:
        ai_usage_tracker.record_llm_usage(
            answer_id, "anthropic" if provider == "claude" else "openai", model, response
        )
    except Exception:
        logger.exception("[pronunciation_reference] failed to record ai usage")
    return response.content.strip()


# Xen Claude/OpenAI/Claude cho 3 lượt sinh reference -- cùng lý do case (5): tăng tính độc lập
# thật giữa 3 candidate (không chỉ khác nhờ temperature=0.7 của CÙNG 1 model), và giảm tải
# OpenAI mỗi turn. Đây là tác vụ SINH VĂN BẢN (không phải chấm điểm JSON), nên retry/fallback
# vẫn dùng chung call_with_retry_and_fallback (nhận closure 0-tham số) từ confidence_utils.
_REFERENCE_PROVIDERS: tuple[tuple[str, str], ...] = (
    ("claude", "openai"),
    ("openai", "claude"),
    ("claude", "openai"),
)


def _build_reference_round(
    transcript: str, question: Optional[QuestionContext], index: int, answer_id: Optional[str]
) -> str:
    primary, fallback = _REFERENCE_PROVIDERS[index]
    return call_with_retry_and_fallback(
        lambda: build_pronunciation_reference(transcript, question, provider=primary, answer_id=answer_id),
        lambda: build_pronunciation_reference(transcript, question, provider=fallback, answer_id=answer_id),
    )


def build_pronunciation_reference_consensus(
    transcript: str,
    question: Optional[QuestionContext] = None,
    *,
    answer_id: Optional[str] = None,
) -> tuple[str, Optional[float]]:
    """Sinh ba reference độc lập song song (xen Claude/OpenAI, mỗi lượt tự retry+fallback) rồi
    chọn medoid và tính C_ref."""
    if not transcript or not transcript.strip():
        return transcript, None

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(_build_reference_round, transcript, question, index, answer_id)
            for index in range(3)
        ]
        references = [future.result() for future in futures]

    reference, confidence, stability, drift = compute_reference_confidence(
        transcript,
        references,
    )
    logger.info(
        "[eval:pronunciation] reference consensus stability=%s drift=%s confidence=%s",
        stability,
        drift,
        confidence,
    )
    return reference, confidence
