"""Helper functions for the coherence evaluation node.

Deliberately duplicated (not shared) across CoherenceEvalNode, GrammarEvalNode,
and LexicalEvalNode: each node owns its own copy so utils/ stays limited to
helpers that are genuinely generic, and each node's prompt-building logic can
evolve independently.

Transcript priority: transcribed_text > corrected_transcript > reference_text (fallback).
"""

import json
import logging
from typing import Any, Callable, Dict, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from node.state_models import SpeakingInput
from schemas.enums import SpeakingMode
from schemas.scoring import CriterionScore

logger = logging.getLogger(__name__)


def select_text_for_language_scoring(
    speaking_input: SpeakingInput,
) -> Tuple[Optional[str], str]:
    """Pick the best transcript for LLM-based language scoring.

    This selects the student-only text that is actually graded.
    conversation_transcript (the timestamped AI/User dialogue) is added
    separately as additional context in coherence_eval_node_config's
    build_user_prompt -- it is context only, never the graded text itself.

    Returns:
        (text, source) where source is one of:
          - "transcribed_text"
          - "corrected_transcript"
          - "reference_text_fallback"
          - "missing"
    """
    if speaking_input.transcribed_text:
        return speaking_input.transcribed_text, "transcribed_text"

    if speaking_input.corrected_transcript:
        return speaking_input.corrected_transcript, "corrected_transcript"

    if speaking_input.reference_text:
        return speaking_input.reference_text, "reference_text_fallback"

    return None, "missing"


def build_scoring_metadata(
    source: str,
    mode: Optional[str],
) -> dict:
    """Build metadata dict for the selected transcript source.

    Includes a diagnostic flag when the source is a reference_text fallback,
    meaning the score should NOT be treated as a real student assessment.
    """
    meta: dict = {"language_scoring_text_source": source}

    if source == "reference_text_fallback":
        meta["language_scoring_note"] = (
            "No transcribed_text or corrected_transcript available. "
            "Fell back to reference_text which may be a model answer. "
            "Score is diagnostic only, not an official assessment."
        )
        meta["language_scoring_status"] = "diagnostic_only"

    if mode == SpeakingMode.SCRIPTED and source != "transcribed_text":
        meta["language_scoring_note"] = (
            "Scripted mode: LLM score is diagnostic only. "
            "Official scoring uses Azure pronunciation assessment."
        )
        meta["language_scoring_status"] = "diagnostic_only"

    return meta


def build_question_context(speaking_input: SpeakingInput) -> str:
    """Build question/topic context block for the LLM prompt."""
    parts = []
    q = speaking_input.question
    t = speaking_input.topic

    if q and q.question_text:
        parts.append(f'Question: "{q.question_text}"')
    if q and q.question_type:
        parts.append(f"Question type: {q.question_type}")
    if q and q.difficulty_level:
        parts.append(f"Difficulty: {q.difficulty_level}")
    if q and q.duration_seconds is not None:
        parts.append(f"Expected duration: {q.duration_seconds}s")
    if q and q.min_response_seconds is not None and q.max_response_seconds is not None:
        parts.append(
            f"Expected response length: {q.min_response_seconds}-{q.max_response_seconds}s"
        )
    elif q and q.min_response_seconds is not None:
        parts.append(f"Expected response length: at least {q.min_response_seconds}s")
    elif q and q.max_response_seconds is not None:
        parts.append(f"Expected response length: up to {q.max_response_seconds}s")
    if t and t.topic_name:
        parts.append(f"Topic: {t.topic_name}")
    if t and t.topic_description:
        parts.append(f"Topic description: {t.topic_description}")
    if q and q.evaluation_guide:
        g = q.evaluation_guide
        parts.append("\n## Evaluation Guide")
        if g.expected_content:
            parts.append(f"Expected content: {g.expected_content}")
        if g.key_points:
            parts.append(f"Key points: {g.key_points}")
        if g.acceptable_responses:
            parts.append(f"Acceptable responses: {g.acceptable_responses}")
        if g.off_topic_examples:
            parts.append(f"Off-topic examples: {g.off_topic_examples}")
        if g.scoring_hints:
            parts.append(f"Scoring hints: {g.scoring_hints}")
        if g.common_mistakes:
            parts.append(f"Common mistakes for this question (supporting context, not exhaustive): {g.common_mistakes}")

    return "\n".join(parts) if parts else "No question context provided."


def build_framework_criterion_context(speaking_input: SpeakingInput, criterion_key: str) -> str:
    """Build the scoring-framework block for a criterion, or return empty string."""
    match = next(
        (
            cf
            for cf in (speaking_input.criteria_frameworks or [])
            if cf.criterion_key == criterion_key
        ),
        None,
    )
    if match is None:
        return ""

    lines = ["## Scoring Framework"]
    if match.framework_criterion_name:
        lines.append(f"Framework criterion: {match.framework_criterion_name}")
    if match.framework_criterion_description:
        lines.append(f"Definition: {match.framework_criterion_description}")
    lines.append(
        f"Score range for this criterion: {match.rubric_min_score}-{match.rubric_max_score}"
    )

    for band in sorted(match.bands, key=lambda b: b.score_min):
        line = f"- {band.code} ({band.score_min}-{band.score_max})"
        if band.label:
            line += f" [{band.label}]"
        if band.descriptor:
            line += f": {band.descriptor}"
        lines.append(line)
        if band.positive_signals:
            lines.append(f"  Positive signals: {', '.join(band.positive_signals)}")
        if band.negative_signals:
            lines.append(f"  Negative signals: {', '.join(band.negative_signals)}")

    return "\n".join(lines)


def call_llm(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Call LLM and parse JSON response."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    content = response.content.strip()

    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()

    return json.loads(content)


def merge_criterion(pronunciation_result: Any, criterion_key: str, llm_response: Dict[str, Any]) -> Any:
    """Merge LLM score into pronunciation_result.criteria.<criterion_key>."""
    criterion = CriterionScore(
        score=llm_response["score"],
        subscores=llm_response.get("subscores", {}),
        note=llm_response.get("note", "Evaluated by LLM based on transcript analysis."),
    )
    setattr(pronunciation_result.criteria, criterion_key, criterion)
    return pronunciation_result


def run_eval_node(
    state: Dict[str, Any],
    criterion_key: str,
    system_prompt: str,
    build_user_prompt_fn: Callable[[SpeakingInput, str], str],
    node_name: str,
) -> Dict[str, Any]:
    """Run the coherence eval node: guards, transcript selection, LLM call, merge, error handling."""
    speaking_input = state.get("speaking_input")
    pronunciation_result = state.get("pronunciation_result")
    answer_id = getattr(speaking_input, "answer_id", None)
    turn_order = (state.get("metadata") or {}).get("turn_order")

    if speaking_input is None:
        return {**state, "status": "error", "error": f"speaking_input is required for {node_name}_eval_node"}

    if pronunciation_result is None:
        return {**state, "status": "error", "error": "pronunciation_result is required. Run pronunciation_eval first."}

    transcript, source = select_text_for_language_scoring(speaking_input)

    if transcript is None:
        return {**state, "status": "error", "error": f"No transcript available for {node_name} evaluation"}

    scoring_meta = build_scoring_metadata(source, speaking_input.mode)

    logger.info("[eval:%s] calling LLM answer_id=%s turn=%s", node_name, answer_id, turn_order)

    try:
        user_prompt = build_user_prompt_fn(speaking_input, transcript)
        llm_response = call_llm(system_prompt, user_prompt)
        updated_result = merge_criterion(pronunciation_result, criterion_key, llm_response)

        existing_meta = state.get("metadata") or {}
        merged_meta = {**existing_meta, **scoring_meta}

        logger.info(
            "[eval:%s] done answer_id=%s turn=%s score=%s",
            node_name, answer_id, turn_order, llm_response.get("score"),
        )
        return {**state, "pronunciation_result": updated_result, "metadata": merged_meta, "status": "completed", "error": None}

    except json.JSONDecodeError as exc:
        logger.exception("[eval:%s] LLM returned invalid JSON answer_id=%s turn=%s", node_name, answer_id, turn_order)
        return {**state, "status": "error", "error": f"LLM returned invalid JSON: {str(exc)}"}

    except Exception as exc:
        logger.exception("[eval:%s] failed answer_id=%s turn=%s", node_name, answer_id, turn_order)
        return {**state, "status": "error", "error": f"{node_name} evaluation failed: {str(exc)}"}
