"""Shared helper to build question/topic context block for LLM prompts."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from node.state_models import SpeakingInput


def build_question_context(speaking_input: "SpeakingInput") -> str:
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
