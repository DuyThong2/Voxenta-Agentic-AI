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
from node.evalGraph.StartNode.start_node_config import start_node
from node.evalGraph.ValidityNode.validity_node_config import validity_node


def route_after_validity(state: GraphState) -> str:
    """Route to END if validity rejects, otherwise continue to pronunciation_eval."""
    validity = state.get("validity")
    if validity and getattr(validity, "action", None) == "reject_or_zero":
        return "end"
    return "continue"


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

    g.add_conditional_edges(
        "strict_validity_check",
        route_after_validity,
        {
            "end": END,
            "continue": "pronunciation_eval",
        },
    )
    g.add_conditional_edges(
        "pronunciation_eval",
        route_on_error,
        {
            "end": END,
            "continue": "answer_length_analysis",
        },
    )
    g.add_edge("answer_length_analysis", "coherence_eval")
    g.add_edge("coherence_eval", "lexical_eval")
    g.add_edge("lexical_eval", "grammar_eval")
    g.add_edge("grammar_eval", END)

    if checkpointer is not None:
        return g.compile(checkpointer=checkpointer)

    return g.compile()
