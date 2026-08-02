from enum import Enum

from pydantic import BaseModel, Field, create_model

# Danh mục MẶC ĐỊNH -- chỉ dùng khi Java không gửi `dimensions` xuống (ví dụ gọi tay để thử).
# Nguồn sự thật là bảng interest_dimension bên vox: SYSTEM_ADMIN thêm chiều mới thì Java gửi
# kèm trong request, Python dựng lại schema theo đó (xem build_quiz_batch_model). Nhờ vậy thêm
# chiều không phải deploy lại service này.
DEFAULT_INTEREST_DIMENSIONS = [
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
    # Danh mục chiều hiện hành do Java gửi xuống; rỗng thì dùng DEFAULT_INTEREST_DIMENSIONS.
    dimensions: list[str] = Field(default_factory=list)

    def effective_dimensions(self) -> list[str]:
        return self.dimensions or DEFAULT_INTEREST_DIMENSIONS


class GeneratedQuizItem(BaseModel):
    """Kiểu 'mở' dùng cho đường đọc/khai báo tĩnh. Đường SINH dùng model dựng động từ
    build_quiz_batch_model để ràng buộc dimension ngay lúc decode."""

    dimension_per_statement: list[str] = Field(min_length=3, max_length=3)
    statements: list[str] = Field(min_length=3, max_length=3)
    desirability_check: str = Field(min_length=1)


class InterestQuizItemBatch(BaseModel):
    items: list[GeneratedQuizItem] = Field(max_length=7)


def build_quiz_batch_model(dimensions: list[str]) -> type[BaseModel]:
    """Dựng schema có enum dimension ĐÚNG theo danh mục hiện hành.

    Vì sao không để `list[str]` rồi lọc hậu kiểm: structured output của OpenAI dùng enum trong
    schema để ràng buộc ngay lúc sinh token -- model KHÔNG THỂ trả về chiều không tồn tại. Bỏ
    ràng buộc đó đi thì phải chấp nhận model bịa chiều rồi vứt cả item, tốn thêm lượt gọi.
    Dựng động giữ nguyên bảo đảm đó mà vẫn cho phép admin thêm chiều lúc chạy.
    """
    dimension_enum = Enum(  # type: ignore[misc]
        "DynamicInterestDimension",
        {code: code for code in dimensions},
        type=str,
    )
    item_model = create_model(
        "DynamicGeneratedQuizItem",
        dimension_per_statement=(
            list[dimension_enum],  # type: ignore[valid-type]
            Field(min_length=3, max_length=3),
        ),
        statements=(list[str], Field(min_length=3, max_length=3)),
        desirability_check=(str, Field(min_length=1)),
    )
    return create_model(
        "DynamicInterestQuizItemBatch",
        items=(list[item_model], Field(max_length=7)),  # type: ignore[valid-type]
    )
