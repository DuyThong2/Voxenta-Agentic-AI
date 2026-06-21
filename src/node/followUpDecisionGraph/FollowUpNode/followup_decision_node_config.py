import json
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from node.followUpDecisionGraph.constants import MAX_TURNS
from node.followUpDecisionGraph.FollowUpNode.followup_decision_prompt import SYSTEM_PROMPT
from schemas.evaluation_event import EvaluationGuideInput
from node.state_models import QuestionContext


def _question_attr(question: QuestionContext | Dict[str, Any] | None, key: str) -> Any:
    if question is None:
        return None
    if isinstance(question, dict):
        return question.get(key)
    return getattr(question, key, None)


def _format_guide(guide: EvaluationGuideInput | Dict[str, Any] | None) -> str:
    if guide is None:
        return "No evaluation guide provided."

    parts: List[str] = []
    expected_content = guide.get("expected_content") if isinstance(guide, dict) else guide.expected_content
    key_points = guide.get("key_points") if isinstance(guide, dict) else guide.key_points
    scoring_hints = guide.get("scoring_hints") if isinstance(guide, dict) else guide.scoring_hints
    if expected_content:
        parts.append(f"Expected content: {expected_content}")
    if key_points:
        parts.append(f"Key points: {key_points}")
    if scoring_hints:
        parts.append(f"Scoring hints: {scoring_hints}")
    return "\n".join(parts) if parts else "No evaluation guide provided."


def _format_history(turns: List[Dict[str, Any]]) -> str:
    if not turns:
        return "No previous turns."

    lines: List[str] = []
    for turn in turns:
        lines.append(
            f"Turn {turn['turn_order']} ({turn['turn_type']}): "
            f"prompt={turn.get('prompt_text') or ''} "
            f"transcript={turn.get('transcript') or ''} "
            f"word_count={turn.get('word_count') or 0}"
        )
    return "\n".join(lines)


def _format_question(question: QuestionContext | Dict[str, Any] | None) -> str:
    if question is None:
        return "No question context provided."

    parts: List[str] = []
    question_text = _question_attr(question, "question_text")
    question_type = _question_attr(question, "question_type")
    duration_seconds = _question_attr(question, "duration_seconds")
    min_response_seconds = _question_attr(question, "min_response_seconds")
    max_response_seconds = _question_attr(question, "max_response_seconds")

    if question_text:
        parts.append(f'Question: "{question_text}"')
    if question_type:
        parts.append(f"Question type: {question_type}")
    if duration_seconds is not None:
        parts.append(f"Expected duration: {duration_seconds}s")
    if min_response_seconds is not None and max_response_seconds is not None:
        parts.append(f"Expected response length: {min_response_seconds}-{max_response_seconds}s")
    elif min_response_seconds is not None:
        parts.append(f"Expected response length: at least {min_response_seconds}s")
    elif max_response_seconds is not None:
        parts.append(f"Expected response length: up to {max_response_seconds}s")

    return "\n".join(parts) if parts else "No question context provided."


def _split_turn_history(turns: List[Dict[str, Any]], current_turn: Dict[str, Any]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if turns and turns[-1].get("turn_order") == current_turn.get("turn_order"):
        return turns[:-1], turns[-1]
    return turns, current_turn


def _build_prompt(state: Dict[str, Any]) -> str:
    current_turn = state["current_turn"]
    all_turns = list(state.get("turns", []))
    history, latest_turn = _split_turn_history(all_turns, current_turn)
    question = state.get("question")
    guide = _question_attr(question, "evaluation_guide")
    signals = state.get("signals") or {}

    return (
        "## Question Context\n"
        f"{_format_question(question)}\n\n"
        "## Evaluation Guide\n"
        f"{_format_guide(guide)}\n\n"
        "## Turn Signals\n"
        f"Expected min words: {signals.get('expected_min_words')}\n"
        f"Cumulative word count: {signals.get('cumulative_word_count')}\n"
        f"Length sufficient: {signals.get('length_sufficient')}\n\n"
        "## Previous Turns\n"
        f"{_format_history(history)}\n\n"
        "## Current Turn\n"
        f"Turn {latest_turn.get('turn_order')} ({latest_turn.get('turn_type')}):\n"
        f"Prompt: {latest_turn.get('prompt_text') or ''}\n"
        f"Transcript: {latest_turn.get('transcript') or ''}\n"
        f"Word count: {latest_turn.get('word_count') or 0}\n\n"
        "Decide whether one more follow-up is needed. If you continue, the next prompt must be tied to a concrete detail from the latest transcript."
    )


def followup_decision_node(state: Dict[str, Any]) -> Dict[str, Any]:
    current_turn = state.get("current_turn")
    if current_turn is None:
        return {
            **state,
            "status": "error",
            "error": "current_turn is required for followup_decision_node",
        }

    signals = state.get("signals") or {}
    if signals.get("hard_stop"):
        return {
            **state,
            "status": "completed",
            "decision": {
                "should_continue": False,
                "next_prompt_text": None,
                "reason": signals.get("hard_stop_reason", ""),
            },
        }

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=_build_prompt(state)),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        if content.startswith("```"):
            lines = [line for line in content.splitlines() if not line.strip().startswith("```")]
            content = "\n".join(lines).strip()
        decision = json.loads(content)
    except Exception as exc:
        return {
            **state,
            "status": "completed",
            "error": f"Follow-up decision failed: {exc}",
            "decision": {
                "should_continue": False,
                "next_prompt_text": None,
                "reason": "decision_fallback",
            },
        }

    next_prompt_text = decision.get("next_prompt_text")
    should_continue = bool(decision.get("should_continue"))
    reason = decision.get("reason", "")

    if (state.get("turn_order") or current_turn.get("turn_order") or 0) >= MAX_TURNS:
        should_continue = False
        next_prompt_text = None
        reason = "max_turns_reached"

    if should_continue and not str(next_prompt_text or "").strip():
        should_continue = False
        next_prompt_text = None
        reason = "decision_fallback"

    if not should_continue:
        next_prompt_text = None

    return {
        **state,
        "status": "completed",
        "decision": {
            "should_continue": should_continue,
            "next_prompt_text": next_prompt_text,
            "reason": reason,
        },
    }
