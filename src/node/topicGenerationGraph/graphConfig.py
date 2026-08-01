from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from node.topicGenerationGraph.GraphState import TopicGenerationState
from node.topicGenerationGraph.KeywordExtractionNode import (
    keyword_extraction_node,
)
from node.topicGenerationGraph.TopicDedupeNode import topic_dedupe_node
from node.topicGenerationGraph.TopicProposalNode import topic_proposal_node
from schemas.topic_generation import TopicProposalBatch, TopicProposalRequest


class TopicGenerationGraph:
    def __init__(self, client: OpenAI | None = None) -> None:
        self.client = client or OpenAI()
        graph = StateGraph(TopicGenerationState)
        graph.add_node("keyword_extraction", keyword_extraction_node)
        graph.add_node(
            "topic_proposal",
            lambda state: topic_proposal_node(state, self.client),
        )
        graph.add_node(
            "topic_dedupe",
            lambda state: topic_dedupe_node(state, self.client),
        )
        graph.add_edge(START, "keyword_extraction")
        graph.add_edge("keyword_extraction", "topic_proposal")
        graph.add_edge("topic_proposal", "topic_dedupe")
        graph.add_edge("topic_dedupe", END)
        self.compiled = graph.compile()

    def invoke(self, request: TopicProposalRequest) -> TopicProposalBatch:
        state = self.compiled.invoke({"request": request, "proposals": []})
        return TopicProposalBatch(proposals=state["proposals"])
