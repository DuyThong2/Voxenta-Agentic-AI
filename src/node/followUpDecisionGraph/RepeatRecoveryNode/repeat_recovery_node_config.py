import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from node.followUpDecisionGraph.RepeatRecoveryNode.repeat_recovery_node_helper import (
    PARAPHRASE_PREFIX,
    build_repeat_text,
    format_history,
    question_attr,
    state_without_turns,
)
from node.followUpDecisionGraph.RepeatRecoveryNode.repeat_recovery_node_prompt import (
    SYSTEM_PROMPT,
)


def _build_prompt(state: Dict[str, Any]) -> str:
    current_turn = state["current_turn"]
    turns = list(state.get("turns", []))
    question = state.get("question")
    active_prompt_text = state.get("active_prompt_text") or current_turn.get("prompt_text")
    transcript = current_turn.get("transcript") or ""
    signals = state.get("signals") or {}

    clarification_count = sum(
        1
        for turn in turns
        if str(turn.get("decision_reason") or "").startswith("clarification_")
    )
    respectful_reminder_count = sum(
        1
        for turn in turns
        if str(turn.get("decision_reason") or "") == "clarification_respectful_reminder"
    )

    return (
        "## Original Question\n"
        f"{question_attr(question, 'question_text') or ''}\n\n"
        "## Current Active Prompt\n"
        f"{active_prompt_text or ''}\n\n"
        "## Previous Turns\n"
        f"{format_history(turns)}\n\n"
        "## Current Student Turn\n"
        f"Transcript: {transcript}\n"
        f"Turn type: {current_turn.get('turn_type')}\n"
        f"Word count: {current_turn.get('word_count') or 0}\n\n"
        "## Turn Signals\n"
        f"No meaningful speech: {signals.get('no_meaningful_speech')}\n"
        f"Length sufficient: {signals.get('length_sufficient')}\n"
        f"Clarification attempts already seen: {clarification_count}\n"
        f"Respectful reminders already given: {respectful_reminder_count}\n\n"
        "Decide whether to repair the latest prompt, move on because the student is uncooperative/refusing,"
        " or pass this turn to the normal follow-up decision."
    )


def repeat_recovery_node(state: Dict[str, Any]) -> Dict[str, Any]:
    current_turn = state.get("current_turn")
    if current_turn is None:
        return {
            **state_without_turns(state),
            "status": "error",
            "error": "current_turn is required for repeat_recovery_node",
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
        repair = json.loads(content)
    except Exception as exc:
        return {
            **state_without_turns(state),
            "status": "completed",
            "edge_case_handled": False,
            "error": f"Repeat recovery failed: {exc}",
        }

    action = str(repair.get("action") or "").strip()
    spoken_text = str(repair.get("spoken_text") or "").strip() or None
    active_prompt_text = str(repair.get("active_prompt_text") or "").strip() or None
    reason = str(repair.get("reason") or "").strip() or "clarification_continue"

    current_active_prompt = state.get("active_prompt_text") or current_turn.get("prompt_text")

    if action == "repeat_latest_prompt":
        return {
            **state_without_turns(state),
            "status": "completed",
            "edge_case_handled": True,
            "decision": {
                "should_continue": True,
                "next_prompt_text": spoken_text or build_repeat_text(current_active_prompt, state.get("question")),
                "active_prompt_text": active_prompt_text or current_active_prompt,
                "reason": "clarification_repeat_latest_prompt",
            },
        }

    if action == "paraphrase_latest_prompt":
        paraphrased_prompt = active_prompt_text or spoken_text or current_active_prompt
        spoken = spoken_text or (
            f"{PARAPHRASE_PREFIX} {paraphrased_prompt}" if paraphrased_prompt else PARAPHRASE_PREFIX
        )
        return {
            **state_without_turns(state),
            "status": "completed",
            "edge_case_handled": True,
            "decision": {
                "should_continue": True,
                "next_prompt_text": spoken,
                "active_prompt_text": paraphrased_prompt,
                "reason": "clarification_paraphrase_latest_prompt",
            },
        }

    if action == "encourage_best_effort":
        return {
            **state_without_turns(state),
            "status": "completed",
            "edge_case_handled": True,
            "decision": {
                "should_continue": True,
                "next_prompt_text": spoken_text or "Please answer as best you can from what you heard.",
                "active_prompt_text": current_active_prompt,
                "reason": "clarification_best_effort",
            },
        }

    if action == "remind_respectfully":
        return {
            **state_without_turns(state),
            "status": "completed",
            "edge_case_handled": True,
            "decision": {
                "should_continue": True,
                "next_prompt_text": spoken_text or "Please answer respectfully and do your best with this question.",
                "active_prompt_text": current_active_prompt,
                "reason": "clarification_respectful_reminder",
            },
        }

    if action == "move_on":
        move_on_reason = "clarification_move_on"
        if "uncooperative" in reason.lower() or "refus" in reason.lower() or "non-cooper" in reason.lower():
            move_on_reason = "clarification_uncooperative_move_on"
        return {
            **state_without_turns(state),
            "status": "completed",
            "edge_case_handled": True,
            "decision": {
                "should_continue": False,
                "next_prompt_text": None,
                "active_prompt_text": None,
                "reason": move_on_reason,
            },
        }

    return {
        **state_without_turns(state),
        "status": "completed",
        "edge_case_handled": False,
        "decision": None,
        "error": None,
        "reason": reason,
    }
