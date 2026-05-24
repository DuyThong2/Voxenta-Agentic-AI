"""Build and compile the LangGraph state graph."""

from langgraph.graph import END, START, StateGraph

from node.GraphState import GraphState
from node.PronunciationNode.pronunciation_eval_node_config import (
    pronunciation_eval_node,
)


def build_graph(checkpointer=None):
    g = StateGraph(GraphState)

    g.add_node("pronunciation_eval", pronunciation_eval_node)

    g.add_edge(START, "pronunciation_eval")
    g.add_edge("pronunciation_eval", END)

    if checkpointer is not None:
        return g.compile(checkpointer=checkpointer)

    return g.compile()