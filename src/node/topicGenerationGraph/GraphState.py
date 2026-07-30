from typing import TypedDict

from schemas.topic_generation import (
    TopicProposal,
    TopicProposalRequest,
)


class TopicGenerationState(TypedDict, total=False):
    request: TopicProposalRequest
    proposals: list[TopicProposal]
