from typing import Any, Dict, List

from node.state_models import QuestionContext

PARAPHRASE_PREFIX = "Sure, let me put that more simply:"
DECLINE_REPAIR_PREFIX = "That's okay. Let me rephrase it a little:"


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


def count_decline_repair(turns: List[Dict[str, Any]]) -> int:
    return sum(1 for turn in turns if str(turn.get("decision_reason") or "") == "decline_repair")


def count_uncooperative_warning(turns: List[Dict[str, Any]]) -> int:
    return sum(1 for turn in turns if str(turn.get("decision_reason") or "") == "remind_respectfully")
