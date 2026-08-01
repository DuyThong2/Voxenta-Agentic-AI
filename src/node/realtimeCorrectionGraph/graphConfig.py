from langgraph.graph import END, START, StateGraph

from node.realtimeCorrectionGraph.GraphState import RealtimeCorrectionGraphState
from node.realtimeCorrectionGraph.LightCorrectionNode.light_correction_node_config import (
    light_correction_node,
)
from node.realtimeCorrectionGraph.PronunciationNode.pronunciation_node_config import (
    pronunciation_node,
)


def merge_correction_node(state: RealtimeCorrectionGraphState) -> dict:
    # NOTE: pronunciation_node and light_correction_node run in PARALLEL, fanned out
    # straight from START (mirrors followUpDecisionGraph/graphConfig.py's
    # prepare_turn_signals -> repeat_recovery + followup_decision pattern). Return only
    # the keys this node actually sets -- do not spread `state` back out, or the two
    # parallel branches' shared LastValue channels (transcript/audio_path/language) will
    # collide in the same superstep the same way the NOTE in
    # followUpDecisionGraph/FollowUpNode/followup_decision_node_config.py warns about.
    return {
        "corrections": state.get("light_corrections") or [],
        "status": "completed",
        "error": None,
    }


def build_realtime_correction_graph():
    """Stateless per-call graph (no checkpointer) -- the caller (PracticeAttemptConnection)
    already has the transcript/audio_path in hand for this turn and doesn't need resume
    semantics for a feedback pass that either lands or doesn't."""
    g = StateGraph(RealtimeCorrectionGraphState)
    g.add_node("pronunciation", pronunciation_node)
    g.add_node("light_correction", light_correction_node)
    g.add_node("merge_correction", merge_correction_node)

    g.add_edge(START, "pronunciation")
    g.add_edge(START, "light_correction")
    g.add_edge("pronunciation", "merge_correction")
    g.add_edge("light_correction", "merge_correction")
    g.add_edge("merge_correction", END)

    return g.compile()
