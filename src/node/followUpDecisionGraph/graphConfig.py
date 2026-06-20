from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from node.followUpDecisionGraph.FollowUpNode.followup_decision_node_config import (
    followup_decision_node,
)
from node.followUpDecisionGraph.GraphState import FollowUpGraphState
from utils.speech_client import transcribe


def transcribe_turn_node(state: FollowUpGraphState) -> Dict[str, Any]:
    audio_path = state.get("audio_path")
    if not audio_path:
        return {**state, "status": "error", "error": "audio_path is required"}

    transcript = transcribe(audio_path, state.get("language", "en-US"))
    if not transcript:
        return {**state, "status": "error", "error": "Audio transcription failed"}

    current_turn = {
        "turn_order": state["current_turn_order"],
        "turn_type": "MAIN" if state["current_turn_order"] == 1 else "FOLLOWUP",
        "prompt_text": state.get("current_prompt_text"),
        "transcript": transcript,
    }

    return {
        **state,
        "status": "processing",
        "follow_up_count": max(0, state["current_turn_order"] - 1),
        "current_turn": current_turn,
    }


def route_on_error(state: FollowUpGraphState) -> str:
    if state.get("status") == "error":
        return "end"
    return "continue"


def build_followup_graph(checkpointer=None):
    g = StateGraph(FollowUpGraphState)
    g.add_node("transcribe_turn", transcribe_turn_node)
    g.add_node("followup_decision", followup_decision_node)

    g.add_edge(START, "transcribe_turn")
    g.add_conditional_edges(
        "transcribe_turn",
        route_on_error,
        {
            "end": END,
            "continue": "followup_decision",
        },
    )
    g.add_edge("followup_decision", END)

    if checkpointer is not None:
        return g.compile(checkpointer=checkpointer)
    return g.compile()
