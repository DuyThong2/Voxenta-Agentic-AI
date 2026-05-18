"""Start node: receives user message and generates a response."""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from node.GraphState import GraphState
from node.StartNode.start_node_prompt import SYSTEM_PROMPT


def start_node(state: GraphState) -> dict:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}
