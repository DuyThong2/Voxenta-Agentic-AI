import asyncio
import json
import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from infra.alert_client import push_alert
from node.followUpDecisionGraph.RepeatRecoveryNode.repeat_recovery_node_helper import (
    build_decline_repair_text,
    build_offtopic_redirect_text,
    build_paraphrase_text,
    build_wrong_language_redirect_text,
    count_engagement_violations,
    format_history,
    question_attr,
)
from node.followUpDecisionGraph.RepeatRecoveryNode.repeat_recovery_node_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_REMIND_PREFIX = "We need a respectful answer to continue. Let's try this question once more:"
_ALLOWED_ACTIONS = {
    "continue_normal_followup",
    "clarify_prompt",
    "decline_repair",
    "decline_move_on",
    "remind_respectfully",
    "uncooperative_move_on",
    "redirect_offtopic",
    "offtopic_move_on",
    "redirect_wrong_language",
    "language_move_on",
    "skip_requested",
}

# Reminder-triggering action -> its "budget exhausted" terminal action. These 4 share ONE
# combined budget (see count_engagement_violations) -- max 2 reminders total, 3rd violation
# (any of the 4 types) forces the matching move-on variant.
_FIRST_TIME_TO_MOVE_ON = {
    "decline_repair": "decline_move_on",
    "remind_respectfully": "uncooperative_move_on",
    "redirect_offtopic": "offtopic_move_on",
    "redirect_wrong_language": "language_move_on",
}
_MOVE_ON_TO_FIRST_TIME = {v: k for k, v in _FIRST_TIME_TO_MOVE_ON.items()}
_ENGAGEMENT_VIOLATION_BUDGET = 2


def _format_question(question: Any) -> str:
    if question is None:
        return "No question context provided."

    parts: List[str] = []
    question_text = question_attr(question, "question_text")
    question_type = question_attr(question, "question_type")
    duration_seconds = question_attr(question, "duration_seconds")
    min_response_seconds = question_attr(question, "min_response_seconds")
    max_response_seconds = question_attr(question, "max_response_seconds")

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


def _format_asset(question: Any) -> str:
    if question is None:
        return "No question asset provided."

    asset = question_attr(question, "asset")
    if asset is None:
        return "No question asset provided."

    parts: List[str] = []
    transcript = question_attr(asset, "transcript")
    description = question_attr(asset, "description")
    alt_text = question_attr(asset, "alt_text")
    asset_type = question_attr(asset, "type")

    if asset_type:
        parts.append(f"Asset type: {asset_type}")
    asset_text = transcript or description or alt_text
    if transcript:
        parts.append(f"Asset transcript: {transcript}")
    elif description:
        parts.append(f"Asset description: {description}")
    elif alt_text:
        parts.append(f"Asset alt text: {alt_text}")
    if not asset_text and asset_type:
        parts.append("Asset details: unavailable")
    if asset_text:
        parts.append(
            "(Note: this is objective/factual information about the asset's content, "
            "NOT a model answer or the single correct interpretation. If the question "
            "asks the student to describe feelings, meaning, or their opinion about the "
            "asset, a DIFFERENT interpretation than what's described above is still VALID "
            "as long as it genuinely engages with the asset's actual content and is "
            "reasonably argued -- do not mark it off-topic or incorrect just because it "
            "diverges from this description.)"
        )

    return "\n".join(parts) if parts else "No question asset provided."


def _split_turn_history(turns: List[Dict[str, Any]], current_turn: Dict[str, Any]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if turns and turns[-1].get("turn_order") == current_turn.get("turn_order"):
        return turns[:-1], turns[-1]
    return turns, current_turn


def _build_prompt(state: Dict[str, Any]) -> str:
    current_turn = state["current_turn"]
    question = state.get("question")
    all_turns = list(state.get("turns", []))
    history, latest_turn = _split_turn_history(all_turns, current_turn)
    signals = state.get("signals") or {}
    active_prompt_text = (
        state.get("active_prompt_text")
        or latest_turn.get("prompt_text")
        or question_attr(question, "question_text")
        or ""
    )
    engagement_violation_count = count_engagement_violations(history)

    return (
        "## Question Context\n"
        f"{_format_question(question)}\n\n"
        "## Question Asset\n"
        f"{_format_asset(question)}\n\n"
        "## Active Prompt\n"
        f"{active_prompt_text}\n\n"
        "## Previous Turns\n"
        f"{format_history(history)}\n\n"
        "## Current Turn\n"
        f"Turn {latest_turn.get('turn_order')} ({latest_turn.get('turn_type')}):\n"
        f"Prompt: {latest_turn.get('prompt_text') or ''}\n"
        f"Transcript: {latest_turn.get('transcript') or ''}\n"
        f"Word count: {latest_turn.get('word_count') or 0}\n\n"
        "## State Counters\n"
        f"engagement_violation_count={engagement_violation_count}\n"
        "(counts decline_repair + remind_respectfully + redirect_offtopic + redirect_wrong_language "
        "combined, over the full history of this question -- max 2 before force move-on)\n"
        f"no_meaningful_speech={signals.get('no_meaningful_speech')}\n"
        f"followup_pressure={signals.get('followup_pressure')}\n"
        f"hard_stop={signals.get('hard_stop')}\n\n"
        "Choose exactly one allowed action and return strict JSON only."
    )


def _normalize_action(action: Any) -> str:
    normalized = str(action or "").strip()
    return normalized if normalized in _ALLOWED_ACTIONS else "continue_normal_followup"


def _enforce_escalation(action: str, turns: List[Dict[str, Any]]) -> str:
    violation_count = count_engagement_violations(turns)

    if action in _FIRST_TIME_TO_MOVE_ON and violation_count >= _ENGAGEMENT_VIOLATION_BUDGET:
        return _FIRST_TIME_TO_MOVE_ON[action]
    if action in _MOVE_ON_TO_FIRST_TIME and violation_count < _ENGAGEMENT_VIOLATION_BUDGET:
        return _MOVE_ON_TO_FIRST_TIME[action]
    return action


def _clean_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _schedule_alert(state: Dict[str, Any], *, alert_type: str) -> None:
    room_id = str(state.get("exam_attempt_id") or "").strip()
    if not room_id:
        logger.warning("[repeat_recovery] exam_attempt_id missing, skipping alert %s", alert_type)
        return

    participant_id = str(state.get("candidate_id") or room_id).strip() or room_id
    stream_id = str(state.get("answer_id") or room_id).strip() or room_id

    try:
        asyncio.get_running_loop().create_task(
            push_alert(
                room_id=room_id,
                participant_id=participant_id,
                stream_id=stream_id,
                alert_type=alert_type,
            )
        )
    except RuntimeError:
        logger.warning("[repeat_recovery] no running loop, skipping alert %s", alert_type)


def _decision_payload(
    *,
    should_continue: bool,
    reason: str,
    next_prompt_text: str | None = None,
    active_prompt_text: str | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "should_continue": should_continue,
        "next_prompt_text": next_prompt_text if should_continue else None,
        "reason": reason,
    }
    if should_continue and active_prompt_text:
        payload["active_prompt_text"] = active_prompt_text
    return payload


def _resolve_edge_case_decision(state: Dict[str, Any], llm_decision: Dict[str, Any]) -> Dict[str, Any]:
    # NOTE: this node runs in PARALLEL with followup_decision_node (fan-out
    # from prepare_turn_signals, see graphConfig.py's build_text_followup_graph).
    # Every return here must be a NARROW dict containing only the keys this
    # node actually intends to set (repeat_recovery_* / status / error) --
    # never spread the input `state` back out. FollowUpGraphState's shared
    # keys (status, answer_id, exam_attempt_id, question, signals,
    # current_turn, ...) are plain LastValue channels with no reducer, so if
    # both parallel nodes echo them back in the same superstep, LangGraph
    # raises InvalidUpdateError ("can only receive one value per step") even
    # when both sides write the identical value.
    current_turn = state["current_turn"]
    question = state.get("question")
    prior_turns, latest_turn = _split_turn_history(list(state.get("turns", [])), current_turn)
    current_prompt = (
        _clean_text(state.get("active_prompt_text"))
        or _clean_text(latest_turn.get("prompt_text"))
        or _clean_text(question_attr(question, "question_text"))
    )
    action = _enforce_escalation(_normalize_action(llm_decision.get("action")), prior_turns)
    spoken_text = _clean_text(llm_decision.get("spoken_text"))
    active_prompt_text = _clean_text(llm_decision.get("active_prompt_text")) or current_prompt

    if action == "continue_normal_followup":
        return {
            "repeat_recovery_edge_case_handled": False,
        }

    if action == "clarify_prompt":
        rewritten_prompt = active_prompt_text or current_prompt
        reply_text = spoken_text or build_paraphrase_text(rewritten_prompt, question)
        return {
            "repeat_recovery_edge_case_handled": True,
            "repeat_recovery_decision": _decision_payload(
                should_continue=True,
                next_prompt_text=reply_text,
                active_prompt_text=rewritten_prompt,
                reason="clarify_prompt",
            ),
        }

    if action == "decline_repair":
        rewritten_prompt = active_prompt_text or current_prompt
        reply_text = spoken_text or build_decline_repair_text(rewritten_prompt, question)
        return {
            "repeat_recovery_edge_case_handled": True,
            "repeat_recovery_decision": _decision_payload(
                should_continue=True,
                next_prompt_text=reply_text,
                active_prompt_text=rewritten_prompt,
                reason="decline_repair",
            ),
        }

    if action == "decline_move_on":
        return {
            "repeat_recovery_edge_case_handled": True,
            "repeat_recovery_decision": _decision_payload(
                should_continue=False,
                next_prompt_text=None,
                reason="decline_move_on",
            ),
        }

    if action == "remind_respectfully":
        rewritten_prompt = active_prompt_text or current_prompt
        reply_text = spoken_text or f"{_REMIND_PREFIX} {rewritten_prompt}".strip()
        return {
            "repeat_recovery_edge_case_handled": True,
            "repeat_recovery_decision": _decision_payload(
                should_continue=True,
                next_prompt_text=reply_text,
                active_prompt_text=rewritten_prompt,
                reason="remind_respectfully",
            ),
        }

    if action == "redirect_offtopic":
        rewritten_prompt = active_prompt_text or current_prompt
        reply_text = spoken_text or build_offtopic_redirect_text(rewritten_prompt, question)
        return {
            "repeat_recovery_edge_case_handled": True,
            "repeat_recovery_decision": _decision_payload(
                should_continue=True,
                next_prompt_text=reply_text,
                active_prompt_text=rewritten_prompt,
                reason="redirect_offtopic",
            ),
        }

    if action == "offtopic_move_on":
        return {
            "repeat_recovery_edge_case_handled": True,
            "repeat_recovery_decision": _decision_payload(
                should_continue=False,
                next_prompt_text=None,
                reason="offtopic_move_on",
            ),
        }

    if action == "redirect_wrong_language":
        rewritten_prompt = active_prompt_text or current_prompt
        reply_text = spoken_text or build_wrong_language_redirect_text(rewritten_prompt, question)
        return {
            "repeat_recovery_edge_case_handled": True,
            "repeat_recovery_decision": _decision_payload(
                should_continue=True,
                next_prompt_text=reply_text,
                active_prompt_text=rewritten_prompt,
                reason="redirect_wrong_language",
            ),
        }

    if action == "language_move_on":
        return {
            "repeat_recovery_edge_case_handled": True,
            "repeat_recovery_decision": _decision_payload(
                should_continue=False,
                next_prompt_text=None,
                reason="language_move_on",
            ),
        }

    if action == "uncooperative_move_on":
        _schedule_alert(state, alert_type="CRITICAL_VIOLATION")
        return {
            "repeat_recovery_edge_case_handled": True,
            "repeat_recovery_decision": _decision_payload(
                should_continue=False,
                next_prompt_text=None,
                reason="uncooperative_move_on",
            ),
        }

    if action == "skip_requested":
        return {
            "repeat_recovery_edge_case_handled": True,
            "repeat_recovery_decision": _decision_payload(
                should_continue=False,
                next_prompt_text=None,
                reason="skip_requested",
            ),
        }

    return {
        "repeat_recovery_edge_case_handled": False,
    }


def repeat_recovery_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # See the NOTE in _resolve_edge_case_decision: every return path in this
    # node must be a narrow dict (no `**state` spread) because this node runs
    # in parallel with followup_decision_node in the same LangGraph superstep.
    current_turn = state.get("current_turn")
    if current_turn is None:
        return {
            "repeat_recovery_edge_case_handled": False,
            "repeat_recovery_error": "current_turn is required for repeat_recovery_node",
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
        logger.warning("[repeat_recovery] LLM decision failed: %s", exc, exc_info=True)
        return {
            "repeat_recovery_edge_case_handled": False,
            "repeat_recovery_error": f"Repeat recovery decision failed: {exc}",
        }

    return _resolve_edge_case_decision(state, decision)
