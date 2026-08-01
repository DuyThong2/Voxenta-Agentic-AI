from fastapi import APIRouter

from vector.practice_question_selection import (
    NeighborQuestionRequest,
    NeighborQuestionResponse,
    SimilarityRequest,
    SimilarityResponse,
    max_similarities,
    neighbor_questions,
)

router = APIRouter(
    prefix="/internal/practice-selection",
    tags=["Practice selection"],
)


@router.post(
    "/question-similarities",
    response_model=SimilarityResponse,
)
async def question_similarities(
    request: SimilarityRequest,
) -> SimilarityResponse:
    return max_similarities(request)


@router.post(
    "/neighbor-questions",
    response_model=NeighborQuestionResponse,
)
async def query_neighbor_questions(
    request: NeighborQuestionRequest,
) -> NeighborQuestionResponse:
    return neighbor_questions(request)
