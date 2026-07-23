"""Builds a pronunciation-alignment reference text from a raw (uncorrected)
transcript -- grounded in the question/evaluation-guide/asset context so
ambiguous or garbled words are resolved toward what the question was actually
about, not a random unrelated word. Used exclusively as pronunciation_eval_node's
reference_text; every other consumer (validity, grammar/lexical/coherence eval,
UI display) keeps reading speaking_input.transcribed_text untouched."""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from node.evalGraph.PronunciationNode.pronunciation_reference_prompt import SYSTEM_PROMPT
from node.state_models.speaking_input import QuestionContext
from utils.confidence_utils import compute_reference_confidence

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


def build_pronunciation_reference(transcript: str, question: Optional[QuestionContext] = None) -> str:
    if not transcript or not transcript.strip():
        return transcript

    llm = ChatOpenAI(model="gpt-5.4", temperature=0.7)

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

    response = llm.invoke(messages)
    return response.content.strip()


def build_pronunciation_reference_consensus(
    transcript: str,
    question: Optional[QuestionContext] = None,
) -> tuple[str, Optional[float]]:
    """Sinh ba reference độc lập song song rồi chọn medoid và tính C_ref."""
    if not transcript or not transcript.strip():
        return transcript, None

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(build_pronunciation_reference, transcript, question)
            for _ in range(3)
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
