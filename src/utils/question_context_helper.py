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
    if t and t.topic_name:
        parts.append(f"Topic: {t.topic_name}")
    if t and t.topic_description:
        parts.append(f"Topic description: {t.topic_description}")

    return "\n".join(parts) if parts else "No question context provided."
