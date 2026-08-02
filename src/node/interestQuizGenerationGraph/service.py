from openai import OpenAI

from node.interestQuizGenerationGraph.constants import MAX_STATEMENT_WORDS, MODEL
from node.interestQuizGenerationGraph.prompt import build_interest_quiz_prompt
from schemas.interest_quiz_generation import (
    GeneratedQuizItem,
    InterestQuizItemBatch,
    InterestQuizItemGenerationRequest,
    build_quiz_batch_model,
)

_client: OpenAI | None = None


def generate_interest_quiz_items(
    request: InterestQuizItemGenerationRequest,
) -> InterestQuizItemBatch:
    response = _openai_client().responses.parse(
        model=MODEL,
        reasoning={"effort": "medium"},
        input=[
            {
                "role": "system",
                "content": (
                    "You write balanced, psychometrically-sound forced-choice interest "
                    "inventory items for Vietnamese high schoolers."
                ),
            },
            {
                "role": "user",
                "content": build_interest_quiz_prompt(request),
            },
        ],
        # Schema dựng theo danh mục chiều Java gửi xuống -> model bị ràng buộc ngay lúc
        # sinh token, không thể trả về chiều không tồn tại trong hệ thống.
        text_format=build_quiz_batch_model(request.effective_dimensions()),
    )
    raw_items = (
        [] if response.output_parsed is None else response.output_parsed.items
    )
    # Model động dùng Enum cho dimension -> đổi về str thuần trước khi trả ra ngoài, để
    # phần còn lại (và Java) chỉ thấy chuỗi như cũ.
    generated = [
        GeneratedQuizItem(
            dimension_per_statement=[
                str(getattr(value, "value", value))
                for value in item.dimension_per_statement
            ],
            statements=list(item.statements),
            desirability_check=item.desirability_check,
        )
        for item in raw_items
    ]
    existing = {_normalize(s) for s in request.existing_statements}
    valid = _filter_structurally_valid(generated, existing)
    return InterestQuizItemBatch(items=valid[: request.max_items])


def _filter_structurally_valid(
    items: list[GeneratedQuizItem],
    existing_statements: set[str],
) -> list[GeneratedQuizItem]:
    """Lưới an toàn tối thiểu THAY THẾ pilot/kiểm định tâm lý học (n=30, Cronbach's alpha,
    CITC, chi-square) mà nghiên cứu yêu cầu trước khi dùng thật -- đó là nghiên cứu con người,
    không tự động hóa được (xem task/implement/13-..., mục 1). Đây CHỈ kiểm tra cấu trúc: đúng
    3 statement/triplet, đúng độ dài, không trùng dimension trong 1 triplet, không trùng
    statement đã có."""
    accepted: list[GeneratedQuizItem] = []
    seen = set(existing_statements)
    for item in items:
        if len(item.statements) != 3 or len(item.dimension_per_statement) != 3:
            continue
        if len(set(item.dimension_per_statement)) != 3:
            continue
        if any(len(statement.split()) > MAX_STATEMENT_WORDS for statement in item.statements):
            continue
        normalized = [_normalize(s) for s in item.statements]
        if any(s in seen for s in normalized):
            continue
        if len(set(normalized)) != 3:
            continue
        seen.update(normalized)
        accepted.append(item)
    return accepted


def _normalize(statement: str) -> str:
    return " ".join(statement.strip().lower().split())


def _openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client
