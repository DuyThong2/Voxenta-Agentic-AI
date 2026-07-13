from typing import Any, Dict, List

from node.state_models import QuestionContext

PARAPHRASE_PREFIX = "Sure, let me put that more simply:"
DECLINE_REPAIR_PREFIX = "That's okay. Let me rephrase it a little:"
OFFTOPIC_REDIRECT_PREFIX = "Let's stay focused on the question -- here it is again:"
WRONG_LANGUAGE_REDIRECT_PREFIX = "Please try answering in English. Here's the question again:"


def question_attr(question: QuestionContext | Dict[str, Any] | None, key: str) -> Any:
    if question is None:
        return None
    if isinstance(question, dict):
        return question.get(key)
    return getattr(question, key, None)


def format_history(turns: List[Dict[str, Any]]) -> str:
    if not turns:
        return "No previous turns."

    lines: List[str] = []
    for turn in turns:
        lines.append(
            f"Turn {turn.get('turn_order')} ({turn.get('turn_type')}): "
            f"prompt={turn.get('prompt_text') or ''} "
            f"transcript={turn.get('transcript') or ''} "
            f"decision_reason={turn.get('decision_reason') or ''}"
        )
    return "\n".join(lines)


def build_paraphrase_text(active_prompt_text: str | None, question: QuestionContext | Dict[str, Any] | None) -> str:
    prompt = str(active_prompt_text or question_attr(question, "question_text") or "").strip()
    return f"{PARAPHRASE_PREFIX} {prompt}".strip() if prompt else PARAPHRASE_PREFIX


def build_decline_repair_text(active_prompt_text: str | None, question: QuestionContext | Dict[str, Any] | None) -> str:
    prompt = str(active_prompt_text or question_attr(question, "question_text") or "").strip()
    return f"{DECLINE_REPAIR_PREFIX} {prompt}".strip() if prompt else DECLINE_REPAIR_PREFIX


def build_offtopic_redirect_text(active_prompt_text: str | None, question: QuestionContext | Dict[str, Any] | None) -> str:
    prompt = str(active_prompt_text or question_attr(question, "question_text") or "").strip()
    return f"{OFFTOPIC_REDIRECT_PREFIX} {prompt}".strip() if prompt else OFFTOPIC_REDIRECT_PREFIX


def build_wrong_language_redirect_text(active_prompt_text: str | None, question: QuestionContext | Dict[str, Any] | None) -> str:
    prompt = str(active_prompt_text or question_attr(question, "question_text") or "").strip()
    return f"{WRONG_LANGUAGE_REDIRECT_PREFIX} {prompt}".strip() if prompt else WRONG_LANGUAGE_REDIRECT_PREFIX


# `decline_repair`, `remind_respectfully`, `redirect_offtopic`, and `redirect_wrong_language`
# each get their OWN reminder budget (see _enforce_escalation in repeat_recovery_node_config.py)
# -- 1 reminder per type before that type's own move-on variant is forced on its next
# occurrence. Types are independent: a 2nd violation of a DIFFERENT type than the 1st still
# gets its own first-time reminder (it hasn't used its budget yet); only a REPEAT of the SAME
# type forces escalation. Counted as a TOTAL over the full turn history (sum, not "reset when
# a compliant turn appears in between") -- a student who alternates between violating and
# complying must not be able to reset a type's budget by giving one clean turn in between.
_ENGAGEMENT_VIOLATION_REASONS = {
    "decline_repair",
    "remind_respectfully",
    "redirect_offtopic",
    "redirect_wrong_language",
}


def count_engagement_violations(turns: List[Dict[str, Any]]) -> int:
    """Total violations across all 4 types combined -- informational only (shown to the
    LLM as context); enforcement itself is per-type, see count_violations_of_type."""
    return sum(1 for turn in turns if str(turn.get("decision_reason") or "") in _ENGAGEMENT_VIOLATION_REASONS)


def count_violations_of_type(turns: List[Dict[str, Any]], reason: str) -> int:
    """How many times this SPECIFIC violation reason has already occurred in the turn
    history -- the count _enforce_escalation actually gates on (each of the 4 reasons has
    its own independent 1-occurrence budget, not a shared pool)."""
    return sum(1 for turn in turns if str(turn.get("decision_reason") or "") == reason)
