from fastapi import APIRouter

from practice_generation.topic_service import (
    TopicProposalBatch,
    TopicProposalRequest,
    TopicIndexRequest,
    index_topic,
    propose_topics,
)

router = APIRouter(prefix="/internal/practice-generation", tags=["Practice generation"])


@router.post("/topics", response_model=TopicProposalBatch)
async def generate_topics(request: TopicProposalRequest) -> TopicProposalBatch:
    return propose_topics(request)


@router.post("/topics/index", status_code=204)
async def upsert_topic_index(request: TopicIndexRequest) -> None:
    index_topic(request)
