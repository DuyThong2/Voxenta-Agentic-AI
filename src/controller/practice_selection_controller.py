from fastapi import APIRouter

from practice_generation.question_selection import (
    SimilarityRequest,
    SimilarityResponse,
    max_similarities,
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
