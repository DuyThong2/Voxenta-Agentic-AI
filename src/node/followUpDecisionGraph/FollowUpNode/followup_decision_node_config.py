import json
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from node.followUpDecisionGraph.FollowUpNode.followup_decision_prompt import SYSTEM_PROMPT
from schemas.evaluation_event import EvaluationGuideInput


def _format_guide(guide: EvaluationGuideInput | None) -> str:
    if guide is None:
        return "No evaluation guide provided."

    parts: List[str] = []
    if guide.expected_content:
        parts.append(f"Expected content: {guide.expected_content}")
    if guide.key_points:
        parts.append(f"Key points: {guide.key_points}")
    if guide.acceptable_responses:
        parts.append(f"Acceptable responses: {guide.acceptable_responses}")
    if guide.off_topic_examples:
        parts.append(f"Off-topic examples: {guide.off_topic_examples}")
    if guide.scoring_hints:
        parts.append(f"Scoring hints: {guide.scoring_hints}")
    return "\n".join(parts) if parts else "No evaluation guide provided."


def _format_history(turns: List[Dict[str, Any]]) -> str:
    if not turns:
        return "No previous turns."

    lines: List[str] = []
    for turn in turns:
        lines.append(
            f"Turn {turn['turn_order']} ({turn['turn_type']}): "
            f"prompt={turn.get('prompt_text') or ''} "
            f"transcript={turn.get('transcript') or ''}"
        )
    return "\n".join(lines)


def _build_prompt(state: Dict[str, Any]) -> str:
    current_turn = state["current_turn"]
    history = state.get("turns", [])
    guide = state.get("evaluation_guide")
    question_text = state.get("question_text") or ""

    return (
        f"Question: {question_text}\n\n"
        f"Evaluation guide:\n{_format_guide(guide)}\n\n"
        f"Conversation history:\n{_format_history(history)}\n\n"
        f"Current turn transcript:\n{current_turn['transcript']}\n\n"
        f"Current turn order: {current_turn['turn_order']}\n"
        "Decide whether to continue."
    )


def followup_decision_node(state: Dict[str, Any]) -> Dict[str, Any]:
    current_turn = state.get("current_turn")
    if current_turn is None:
        return {
            **state,
            "status": "error",
            "error": "current_turn is required for followup_decision_node",
        }

    if current_turn["turn_order"] >= 3:
        return {
            **state,
            "status": "completed",
            "decision": {
                "should_continue": False,
                "next_prompt_text": None,
                "reason": "max_turns_reached",
            },
            "turns": [current_turn],
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
            "status": "error",
            "error": f"Follow-up decision failed: {exc}",
        }

    next_prompt_text = decision.get("next_prompt_text")
    should_continue = bool(decision.get("should_continue"))
    if not should_continue:
        next_prompt_text = None

    return {
        **state,
        "status": "completed",
        "decision": {
            "should_continue": should_continue,
            "next_prompt_text": next_prompt_text,
            "reason": decision.get("reason", ""),
        },
        "turns": [current_turn],
    }
