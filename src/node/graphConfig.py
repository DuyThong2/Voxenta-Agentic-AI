"""Build and compile the LangGraph state graph."""

from langgraph.graph import END, START, StateGraph

from node.GraphState import GraphState
from node.StartNode.start_node_config import start_node


def build_graph(checkpointer=None):
    g = StateGraph(GraphState)
    g.add_node("start", start_node)
    g.add_edge(START, "start")
    g.add_edge("start", END)
    return g.compile(checkpointer=checkpointer) if checkpointer is not None else g.compile()
