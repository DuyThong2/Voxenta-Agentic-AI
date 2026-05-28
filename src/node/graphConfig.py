"""Build and compile the LangGraph state graph."""

from langgraph.graph import END, START, StateGraph

from node.GraphState import GraphState
from node.CorrectionNode.correction_node_config import correction_node
from node.CoherenceEvalNode.coherence_eval_node_config import coherence_eval_node
from node.GrammarEvalNode.grammar_eval_node_config import grammar_eval_node
from node.LexicalEvalNode.lexical_eval_node_config import lexical_eval_node
from node.PronunciationNode.pronunciation_eval_node_config import (
    pronunciation_eval_node,
)
from node.AnswerLengthNode.answer_length_analysis_node_config import (
    answer_length_analysis_node,
)
from node.StartNode.start_node_config import start_node


def build_graph(checkpointer=None):
    g = StateGraph(GraphState)

    g.add_node("start", start_node)
    g.add_node("correction", correction_node)
    g.add_node("pronunciation_eval", pronunciation_eval_node)
    g.add_node("answer_length_analysis", answer_length_analysis_node)
    g.add_node("coherence_eval", coherence_eval_node)
    g.add_node("lexical_eval", lexical_eval_node)
    g.add_node("grammar_eval", grammar_eval_node)

    g.add_edge(START, "start")
    g.add_edge("start", "correction")
    g.add_edge("correction", "pronunciation_eval")
    g.add_edge("pronunciation_eval", "answer_length_analysis")
    g.add_edge("answer_length_analysis", "coherence_eval")
    g.add_edge("coherence_eval", "lexical_eval")
    g.add_edge("lexical_eval", "grammar_eval")
    g.add_edge("grammar_eval", END)

    if checkpointer is not None:
        return g.compile(checkpointer=checkpointer)

    return g.compile()