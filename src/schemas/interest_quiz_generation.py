from typing import Literal

from pydantic import BaseModel, Field

InterestDimension = Literal[
    "ENTERTAINMENT_MEDIA",
    "TECH_GAMING",
    "SPORTS_HEALTH",
    "PEOPLE_SOCIETY",
    "TRAVEL_PLACES",
    "FUTURE_SCIENCE",
]


class InterestQuizItemGenerationRequest(BaseModel):
    # 7 = QUIZ_ITEM_COUNT bên Java (ViewInterestQuizItemsUseCase) -- khớp đúng số câu
    # submitQuiz yêu cầu (5-7 câu trả lời) và độ dài bộ tĩnh gốc trong interest-quiz-seed.json.
    max_items: int = Field(default=7, ge=1, le=7)
    # Statement text already in the bank (static seed + whatever this student already has) --
    # asked not to repeat, not used to bias content (no personalization-by-history, see
    # task/implement/13-quiz-so-thich-sinh-theo-tinh-huong.md mục 0).
    existing_statements: list[str] = Field(default_factory=list)


class GeneratedQuizItem(BaseModel):
    dimension_per_statement: list[InterestDimension] = Field(min_length=3, max_length=3)
    statements: list[str] = Field(min_length=3, max_length=3)
    desirability_check: str = Field(min_length=1)


class InterestQuizItemBatch(BaseModel):
    items: list[GeneratedQuizItem] = Field(max_length=7)
