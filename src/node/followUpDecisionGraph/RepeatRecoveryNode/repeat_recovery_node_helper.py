from typing import Any, Dict, List

from node.state_models import QuestionContext

REPEAT_PREFIX = "No problem, let me say that again:"
PARAPHRASE_PREFIX = "Sure, let me put that more simply:"


def state_without_turns(state: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in state.items() if k != "turns"}


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


def build_repeat_text(active_prompt_text: str | None, question: QuestionContext | Dict[str, Any] | None) -> str:
    prompt = str(active_prompt_text or question_attr(question, "question_text") or "").strip()
    if not prompt:
        return REPEAT_PREFIX
    return f"{REPEAT_PREFIX} {prompt}"
