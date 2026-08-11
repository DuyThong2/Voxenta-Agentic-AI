from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from node.topicGenerationGraph.constants import MAX_PROPOSAL_ROUNDS
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
        graph.add_conditional_edges(
            "topic_dedupe",
            _should_retry,
            {"retry": "topic_proposal", "done": END},
        )
        self.compiled = graph.compile()

    def invoke(self, request: TopicProposalRequest) -> TopicProposalBatch:
        state = self.compiled.invoke(
            {"request": request, "proposals": [], "accepted": [], "round": 0}
        )
        return TopicProposalBatch(proposals=state.get("accepted") or [])


def _should_retry(state: TopicGenerationState) -> str:
    """Co de xuat lai khong, sau khi bo loc trung vua cat bot.

    Ba dieu kien, phai dung CA BA thi moi quay lai:

    1. Con thieu chu de so voi so can -- du roi thi dung, khong sinh thua.
    2. Chua cham tran vong (MAX_PROPOSAL_ROUNDS) -- chan vong lap vo han khi mot vung ngu nghia
       da bao hoa that su va vong nao cung va cham.
    3. Vong vua roi CO va cham -- neu khong va cham ma van thieu thi model da tra ve it hon so
       xin, de xuat lai voi cung mot prompt cung se ra ket qua nhu vay. Thieu dieu kien nay la
       tu chuoc lay MAX_PROPOSAL_ROUNDS luot goi LLM khong loi ich.
    """
    request = state["request"]
    accepted = len(state.get("accepted") or [])
    if accepted >= request.max_proposals:
        return "done"
    if (state.get("round") or 0) >= MAX_PROPOSAL_ROUNDS:
        return "done"
    if not state.get("collisions"):
        return "done"
    return "retry"
