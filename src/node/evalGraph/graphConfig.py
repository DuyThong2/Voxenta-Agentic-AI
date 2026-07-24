"""Build and compile the LangGraph state graph."""

from langgraph.graph import END, START, StateGraph

from node.evalGraph.GraphState import GraphState
from node.evalGraph.LanguageQualityEvalNode.language_quality_eval_node_config import (
    language_quality_eval_node,
)
from node.evalGraph.PronunciationNode.pronunciation_eval_node_config import (
    pronunciation_eval_node,
)
from node.evalGraph.AnswerLengthNode.answer_length_analysis_node_config import (
    answer_length_analysis_node,
)
from node.evalGraph.MergeScoresNode.merge_scores_node_config import merge_scores_node
from node.evalGraph.StartNode.start_node_config import start_node
from node.evalGraph.ValidityNode.validity_node_config import validity_node


def route_after_validity(state: GraphState) -> list[str] | str:
    """Route to END if validity rejects. Otherwise fan out to pronunciation_eval AND
    answer_length_analysis for complete answers.

    Multi-turn fragments only need per-audio pronunciation. Text validity, answer length,
    and language quality run once later against the merged complete answer.
    """
    validity = state.get("validity")
    if validity and getattr(validity, "action", None) == "reject_or_zero":
        return "end"
    metadata = state.get("metadata") or {}
    if metadata.get("validity_scope") == "turn_fragment":
        return "pronunciation_eval"
    return ["pronunciation_eval", "answer_length_analysis"]


def route_after_answer_length(state: GraphState) -> list[str] | str:
    """Language quality needs answer_length_metrics for its criterion-specific caps.

    If answer_length_analysis failed, skip the LLM calls because their result would be
    discarded at merge time.
    """
    metadata = state.get("metadata") or {}
    if metadata.get("answer_length_error"):
        return "merge_scores"
    return "language_quality_eval"


def route_on_error(state: GraphState) -> str:
    if state.get("status") == "error":
        return "end"
    return "continue"


def build_graph(checkpointer=None):
    g = StateGraph(GraphState)

    g.add_node("start", start_node)
    g.add_node("strict_validity_check", validity_node)
    g.add_node("pronunciation_eval", pronunciation_eval_node)
    g.add_node("answer_length_analysis", answer_length_analysis_node)
    g.add_node("language_quality_eval", language_quality_eval_node)
    g.add_node("merge_scores", merge_scores_node)

    # No CorrectionNode: it fed an LLM-rewritten transcript into validity/pronunciation
    # scoring, but the rewrite was frequently wrong (over-corrected disfluencies, mangled
    # code-switched wording) -- both scoring and display now go straight off Azure's own
    # transcription (speech_client.transcribe(), which already handles code-switched
    # Vietnamese via auto-detect + language tagging), no LLM rewrite step in between.
    g.add_edge(START, "start")
    g.add_conditional_edges(
        "start",
        route_on_error,
        {
            "end": END,
            "continue": "strict_validity_check",
        },
    )

    # Fan-out #1: pronunciation_eval (Azure Speech SDK call) and answer_length_analysis
    # (word/sentence counting + optional LLM length judgment) run concurrently -- neither
    # depends on the other's output.
    g.add_conditional_edges(
        "strict_validity_check",
        route_after_validity,
        {
            "end": END,
            "pronunciation_eval": "pronunciation_eval",
            "answer_length_analysis": "answer_length_analysis",
        },
    )

    # Once answer_length_analysis has produced the score caps, one combined node makes three
    # parallel O-C-O requests. Every request returns all three language criteria, while
    # consensus/confidence remains criterion-specific.
    g.add_conditional_edges(
        "answer_length_analysis",
        route_after_answer_length,
        {
            "merge_scores": "merge_scores",
            "language_quality_eval": "language_quality_eval",
        },
    )

    # Fan-in: merge_scores waits for pronunciation plus the combined language result.
    g.add_edge("pronunciation_eval", "merge_scores")
    g.add_edge("language_quality_eval", "merge_scores")
    g.add_edge("merge_scores", END)

    if checkpointer is not None:
        return g.compile(checkpointer=checkpointer)

    return g.compile()
