from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from node.followUpDecisionGraph.FollowUpNode.followup_decision_node_config import (
    followup_decision_node,
)
from node.followUpDecisionGraph.SignalNode.signal_node_config import (
    prepare_turn_signals_node,
)
from node.followUpDecisionGraph.GraphState import FollowUpGraphState
from utils.text_utils import word_count
from utils.speech_client import transcribe


def transcribe_turn_node(state: FollowUpGraphState) -> Dict[str, Any]:
    audio_path = state.get("audio_path")
    if not audio_path:
        return {**state, "status": "error", "error": "audio_path is required"}

    transcript = transcribe(audio_path, state.get("language", "en-US")) or ""

    current_turn = {
        "answer_id": state.get("answer_id"),
        "turn_order": state["turn_order"],
        "turn_type": "MAIN" if state["turn_order"] == 1 else "FOLLOWUP",
        "prompt_text": state.get("prompt_text"),
        "audio_url": state.get("audio_ref"),
        "transcript": transcript,
        "word_count": word_count(transcript),
        "duration_seconds": None,
        "answered_at": None,
    }

    return {
        **state,
        "status": "processing",
        "current_turn": current_turn,
    }


def route_on_error(state: FollowUpGraphState) -> str:
    if state.get("status") == "error":
        return "end"
    return "continue"


def build_followup_graph(checkpointer=None):
    g = StateGraph(FollowUpGraphState)
    g.add_node("transcribe_turn", transcribe_turn_node)
    g.add_node("prepare_turn_signals", prepare_turn_signals_node)
    g.add_node("followup_decision", followup_decision_node)

    g.add_edge(START, "transcribe_turn")
    g.add_conditional_edges(
        "transcribe_turn",
        route_on_error,
        {
            "end": END,
            "continue": "prepare_turn_signals",
        },
    )
    g.add_edge("prepare_turn_signals", "followup_decision")
    g.add_edge("followup_decision", END)

    if checkpointer is not None:
        return g.compile(checkpointer=checkpointer)
    return g.compile()
