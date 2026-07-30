from openai import OpenAI

from node.topicGenerationGraph.constants import MODEL
from node.topicGenerationGraph.GraphState import TopicGenerationState
from node.topicGenerationGraph.TopicProposalNode.topic_proposal_prompt import (
    build_topic_proposal_prompt,
)
from schemas.topic_generation import (
    TopicProposal,
    TopicProposalBatch,
    TopicProposalRequest,
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
        text_format=TopicProposalBatch,
    )
    proposals = (
        []
        if response.output_parsed is None
        else enforce_evidence_caps(
            response.output_parsed.proposals,
            request,
        )
    )
    return {
        "proposals": _deduplicate_evidence_clusters(
            proposals[: request.max_proposals]
        )
    }


def enforce_evidence_caps(
    proposals: list[TopicProposal],
    request: TopicProposalRequest,
) -> list[TopicProposal]:
    evidence_counts = {
        item.keyword.casefold(): item.session_count
        for item in request.keyword_evidence
    }
    ungrounded = 0
    filtered = []
    for proposal in proposals:
        if not proposal.grounded_in_keyword:
            ungrounded += 1
            proposal.confidence = min(proposal.confidence, 0.4)
            if ungrounded > 1:
                continue
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
