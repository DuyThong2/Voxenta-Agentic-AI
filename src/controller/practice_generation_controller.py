from fastapi import APIRouter

from node.questionGenerationGraph.service import (
    generate_questions,
    index_generated_question,
)
from node.topicGenerationGraph.service import index_topic, propose_topics
from schemas.question_generation import (
    QuestionGenerationRequest,
    QuestionGenerationResponse,
    QuestionIndexRequest,
)
from schemas.topic_generation import (
    TopicIndexRequest,
    TopicProposalBatch,
    TopicProposalRequest,
)

router = APIRouter(prefix="/internal/practice-generation", tags=["Practice generation"])


@router.post("/topics", response_model=TopicProposalBatch)
async def generate_topics(request: TopicProposalRequest) -> TopicProposalBatch:
    return propose_topics(request)


@router.post("/topics/index", status_code=204)
async def upsert_topic_index(request: TopicIndexRequest) -> None:
    index_topic(request)


@router.post("/questions", response_model=QuestionGenerationResponse)
async def generate_practice_questions(
    request: QuestionGenerationRequest,
) -> QuestionGenerationResponse:
    return generate_questions(request)


@router.post("/questions/index", status_code=204)
async def upsert_question_index(request: QuestionIndexRequest) -> None:
    index_generated_question(request)
