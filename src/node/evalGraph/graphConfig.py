"""Build and compile the LangGraph state graph."""

from langgraph.graph import END, START, StateGraph

from node.evalGraph.GraphState import GraphState
from node.evalGraph.CoherenceEvalNode.coherence_eval_node_config import coherence_eval_node
from node.evalGraph.GrammarEvalNode.grammar_eval_node_config import grammar_eval_node
from node.evalGraph.LexicalEvalNode.lexical_eval_node_config import lexical_eval_node
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
    answer_length_analysis -- the two are mutually independent (neither reads the
    other's output), so LangGraph runs them concurrently in the same superstep."""
    validity = state.get("validity")
    if validity and getattr(validity, "action", None) == "reject_or_zero":
        return "end"
    return ["pronunciation_eval", "answer_length_analysis"]


def route_after_answer_length(state: GraphState) -> list[str] | str:
    """coherence_eval/lexical_eval/grammar_eval all need answer_length_metrics (for their
    score caps), so they can only fan out AFTER answer_length_analysis completes -- but
    they're mutually independent of EACH OTHER and of pronunciation_eval, so all three run
    concurrently here. If answer_length_analysis itself failed, skip straight to
    merge_scores instead of paying for three LLM calls that would just get discarded."""
    metadata = state.get("metadata") or {}
    if metadata.get("answer_length_error"):
        return "merge_scores"
    return ["coherence_eval", "lexical_eval", "grammar_eval"]


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
    g.add_node("coherence_eval", coherence_eval_node)
    g.add_node("lexical_eval", lexical_eval_node)
    g.add_node("grammar_eval", grammar_eval_node)
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

    # Fan-out #2: once answer_length_analysis has produced the score caps, coherence/
    # lexical/grammar run concurrently -- each is an independent LLM call that only needs
    # the transcript + those caps, not each other's output.
    g.add_conditional_edges(
        "answer_length_analysis",
        route_after_answer_length,
        {
            "merge_scores": "merge_scores",
            "coherence_eval": "coherence_eval",
            "lexical_eval": "lexical_eval",
            "grammar_eval": "grammar_eval",
        },
    )

    # Fan-in: merge_scores waits for pronunciation_eval + whichever of
    # coherence_eval/lexical_eval/grammar_eval actually ran, combines their four
    # independent result keys into one pronunciation_result, and is the only node
    # allowed to set the shared status/error after this point (see
    # MergeScoresNode.merge_scores_node_config's docstring).
    g.add_edge("pronunciation_eval", "merge_scores")
    g.add_edge("coherence_eval", "merge_scores")
    g.add_edge("lexical_eval", "merge_scores")
    g.add_edge("grammar_eval", "merge_scores")
    g.add_edge("merge_scores", END)

    if checkpointer is not None:
        return g.compile(checkpointer=checkpointer)

    return g.compile()
