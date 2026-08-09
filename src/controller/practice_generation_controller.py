import asyncio

from fastapi import APIRouter

from node.interestQuizGenerationGraph.service import generate_interest_quiz_items
from node.questionGenerationGraph.service import (
    generate_questions,
    index_generated_question,
)
from node.topicGenerationGraph.service import index_topic, propose_topics, search_topics
from schemas.interest_quiz_generation import (
    InterestQuizItemBatch,
    InterestQuizItemGenerationRequest,
)
from schemas.question_generation import (
    QuestionGenerationRequest,
    QuestionGenerationResponse,
    QuestionIndexRequest,
)
from schemas.topic_generation import (
    TopicSearchRequest,
    TopicSearchResponse,
    TopicIndexRequest,
    TopicProposalBatch,
    TopicProposalRequest,
)

router = APIRouter(prefix="/internal/practice-generation", tags=["Practice generation"])


@router.post("/topics", response_model=TopicProposalBatch)
async def generate_topics(request: TopicProposalRequest) -> TopicProposalBatch:
    # propose_topics runs a sync LangGraph .invoke() (multi-step LLM calls) --
    # calling it directly here would block the whole event loop for the
    # duration, starving every Kafka consumer's heartbeat (confirmed: caused
    # mass "coordinator dead" rebalancing across all consumer groups).
    return await asyncio.to_thread(propose_topics, request)


@router.post("/topics/search", response_model=TopicSearchResponse)
async def search_topics_endpoint(request: TopicSearchRequest) -> TopicSearchResponse:
    # to_thread: ca nhung embedding lan query Chroma deu la I/O DONG BO. Goi thang trong handler
    # async se khoa event loop, keo theo moi phien realtime dang chay -- dung loi da gap o
    # /turns/archive.
    return await asyncio.to_thread(search_topics, request)


@router.post("/topics/index", status_code=204)
async def upsert_topic_index(request: TopicIndexRequest) -> None:
    index_topic(request)


@router.post("/questions", response_model=QuestionGenerationResponse)
async def generate_practice_questions(
    request: QuestionGenerationRequest,
) -> QuestionGenerationResponse:
    # Same blocking-graph.invoke() issue as /topics above.
    return await asyncio.to_thread(generate_questions, request)


@router.post("/questions/index", status_code=204)
async def upsert_question_index(request: QuestionIndexRequest) -> None:
    index_generated_question(request)


@router.post("/interest-quiz-items", response_model=InterestQuizItemBatch)
async def generate_interest_quiz_items_endpoint(
    request: InterestQuizItemGenerationRequest,
) -> InterestQuizItemBatch:
    # generate_interest_quiz_items uses the sync OpenAI client (not
    # AsyncOpenAI) -- same blocking-event-loop issue as /topics above.
    return await asyncio.to_thread(generate_interest_quiz_items, request)
