from openai import OpenAI

from node.topicGenerationGraph.constants import MODEL
from node.topicGenerationGraph.GraphState import TopicGenerationState
from node.topicGenerationGraph.TopicProposalNode.topic_proposal_prompt import (
    build_topic_proposal_prompt,
)
from schemas.topic_generation import (
    TopicProposal,
    TopicProposalRequest,
    build_topic_batch_model,
)


def topic_proposal_node(
    state: TopicGenerationState,
    client: OpenAI,
) -> dict:
    request = state["request"]
    response = client.responses.parse(
        model=MODEL,
        reasoning={"effort": "low"},
        input=[
            {
                "role": "system",
                "content": (
                    "You design safe, explainable "
                    "speaking-practice topics."
                ),
            },
            {
                "role": "user",
                "content": build_topic_proposal_prompt(request),
            },
        ],
        # Schema dựng theo danh mục chiều Java gửi xuống -- model không thể trả về chiều
        # không tồn tại trong hệ thống.
        text_format=build_topic_batch_model(request.effective_dimensions()),
    )
    proposals = (
        []
        if response.output_parsed is None
        else enforce_evidence_caps(
            [_to_open_proposal(item) for item in response.output_parsed.proposals],
            request,
        )
    )
    return {
        "proposals": _deduplicate_evidence_clusters(
            proposals[: request.max_proposals]
        )
    }


def _to_open_proposal(item) -> TopicProposal:
    """Model động dùng Enum cho interest_dimension -> đổi về str thuần để phần còn lại
    (và Java) chỉ thấy chuỗi như trước."""
    dimension = item.interest_dimension
    return TopicProposal(
        name=item.name,
        interest_dimension=str(getattr(dimension, "value", dimension)),
        curriculum_group=item.curriculum_group,
        confidence=item.confidence,
        reason_text=item.reason_text,
        distinct_from=item.distinct_from,
        evidence_type=item.evidence_type,
        evidence_keywords=list(item.evidence_keywords),
        grounded_in_keyword=item.grounded_in_keyword,
    )


def enforce_evidence_caps(
    proposals: list[TopicProposal],
    request: TopicProposalRequest,
) -> list[TopicProposal]:
    evidence_counts = {
        item.keyword.casefold(): item.session_count
        for item in request.keyword_evidence
    }
    # Trần "tối đa MỘT đề xuất ungrounded" chỉ có nghĩa khi ĐANG CÓ từ khoá quan sát được: nó
    # ngăn LLM bịa thêm chủ đề chẳng liên quan gì tới bằng chứng đang có. Khi danh sách từ khoá
    # RỖNG (đường TopicSuggestionService.synchronousOffers -- đề xuất dựa hoàn toàn vào
    # interest_scores) thì MỌI đề xuất đều nằm ngoài từ khoá, vì không có từ khoá nào cả; giữ
    # nguyên trần ở đây là cắt còn đúng 1 chủ đề mỗi lượt.
    #
    # Trần này nằm SAU khâu sinh, nên tăng max_proposals không cứu được -- LLM trả về 8 rồi bị
    # cắt còn 1 ở đây, im lặng, không log.
    cap_ungrounded = bool(evidence_counts)
    ungrounded = 0
    filtered = []
    for proposal in proposals:
        if not proposal.grounded_in_keyword:
            ungrounded += 1
            if cap_ungrounded:
                proposal.confidence = min(proposal.confidence, 0.4)
                if ungrounded > 1:
                    continue
            else:
                # 0.4 là trần dành cho "bịa thêm ngoài bằng chứng đang có". Ở đây không phải
                # vậy -- bằng chứng là interest_scores -- nên chấm trần theo evidence_type,
                # đúng như luật 5 trong prompt.
                proposal.confidence = min(
                    proposal.confidence,
                    0.6 if proposal.evidence_type == "INTEREST"
                    else 0.7 if proposal.evidence_type == "EXHAUSTED"
                    else 0.4,
                )
        elif proposal.evidence_type in {"KEYWORD", "SEARCH"}:
            counts = [
                evidence_counts[keyword.casefold()]
                for keyword in proposal.evidence_keywords
                if keyword.casefold() in evidence_counts
            ]
            if not counts:
                continue
            sessions = max(counts)
            cap = 0.5 if sessions <= 1 else 0.7 if sessions == 2 else 0.85
            if proposal.evidence_type == "SEARCH":
                if not request.search_keyword or proposal.confidence < 0.9:
                    continue
                cap = 0.95
            proposal.confidence = min(proposal.confidence, cap)
        elif proposal.evidence_type == "INTEREST":
            proposal.confidence = min(proposal.confidence, 0.6)
        elif proposal.evidence_type == "EXHAUSTED":
            proposal.confidence = min(proposal.confidence, 0.7)
        filtered.append(proposal)
    return filtered


def _deduplicate_evidence_clusters(
    proposals: list[TopicProposal],
) -> list[TopicProposal]:
    result = []
    used_keywords: set[str] = set()
    for proposal in proposals:
        keywords = {
            keyword.casefold() for keyword in proposal.evidence_keywords
        }
        if (
            proposal.evidence_type == "KEYWORD"
            and proposal.grounded_in_keyword
            and keywords & used_keywords
        ):
            continue
        result.append(proposal)
        if proposal.evidence_type == "KEYWORD" and proposal.grounded_in_keyword:
            used_keywords.update(keywords)
    return result
