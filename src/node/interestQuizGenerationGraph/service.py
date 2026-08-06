import logging
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from node.interestQuizGenerationGraph.constants import MAX_STATEMENT_WORDS, MODEL
from node.interestQuizGenerationGraph.prompt import (
    build_single_item_prompt,
    dimension_triplets,
)
from schemas.interest_quiz_generation import (
    GeneratedQuizItem,
    InterestQuizItemBatch,
    InterestQuizItemGenerationRequest,
    build_quiz_batch_model,
)

logger = logging.getLogger(__name__)

# Duoi nguong nay khong dung noi mot bo ba chieu KHAC NHAU -- dinh dang quiz ep buoc
# dieu do. Tra rong de Java lui ve kho tinh, thay vi sinh ra item sai dinh dang.
MINIMUM_DIMENSIONS = 3

_client: OpenAI | None = None


def generate_interest_quiz_items(
    request: InterestQuizItemGenerationRequest,
) -> InterestQuizItemBatch:
    """Sinh SONG SONG mỗi item một lượt gọi, thay vì một lượt to cho cả bộ.

    Bản trước gọi đúng một lần và bắt model tự cân bằng chiều rồi trải 14 ứng viên để chọn 7
    -- đo thật 46,8 giây. Java chờ 45s và Tomcat chờ 30s, nên trần NGOÀI bắn trước trần
    TRONG: đoạn dự phòng "Python không sẵn sàng -> dùng kho tĩnh" không bao giờ chạy tới,
    học sinh nhận lỗi thay vì nhận bộ quiz dự phòng.

    Chia nhỏ được vì phần khó duy nhất -- cân bằng chiều giữa các item -- đã được
    dimension_triplets phân công TRƯỚC, không cần model nhìn thấy nhau. Thời gian giờ xấp xỉ
    một item chứ không phải bảy.
    """
    dimensions = request.effective_dimensions()
    if len(dimensions) < MINIMUM_DIMENSIONS:
        logger.warning(
            "[interest_quiz] chi co %d chieu, khong du de dung bo ba -- tra rong",
            len(dimensions),
        )
        return InterestQuizItemBatch(items=[])
    triplets = dimension_triplets(dimensions, request.max_items)
    batch_model = build_quiz_batch_model(dimensions)

    with ThreadPoolExecutor(max_workers=len(triplets)) as pool:
        results = list(pool.map(
            lambda pair: _generate_one(request, pair[1], batch_model, pair[0]),
            enumerate(triplets),
        ))

    generated = [item for item in results if item is not None]
    existing = {_normalize(s) for s in request.existing_statements}
    valid = _filter_structurally_valid(generated, existing)
    return InterestQuizItemBatch(items=valid[: request.max_items])


def _generate_one(
    request: InterestQuizItemGenerationRequest,
    assigned_dimensions: list[str],
    batch_model: type,
    context_index: int,
) -> GeneratedQuizItem | None:
    """Một lượt gọi cho một item. Lỗi một lượt KHÔNG kéo đổ cả bộ -- trả None rồi bỏ qua.

    Sinh song song nghĩa là hỏng lẻ tẻ là chuyện phải chịu được: 6 item vẫn dùng được, còn
    ném lên thì mất trắng cả bảy.
    """
    try:
        response = _openai_client().responses.parse(
            model=MODEL,
            # "low" chu khong "medium": moi luot gio chi phai nghi ra DUNG MOT item, voi bo ba
            # chieu da phan cong san va boi canh da chi dinh -- phan viec nang nhat (can bang
            # chieu giua cac item, trai 14 ung vien roi chon 7) da bi go khoi prompt. Giu
            # "medium" cho mot viec nho nhu vay la tra tien suy luan khong dung cho.
            reasoning={"effort": "low"},
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
                    "content": build_single_item_prompt(
                        request, assigned_dimensions, context_index
                    ),
                },
            ],
            # Vẫn dùng schema batch (model trả về items[]) rồi lấy phần tử đầu: giữ nguyên
            # ràng buộc enum dimension đã có, khỏi dựng thêm một schema thứ hai.
            text_format=batch_model,
        )
    except Exception:
        logger.exception(
            "[interest_quiz] sinh item that bai dimensions=%s", assigned_dimensions
        )
        return None

    raw_items = [] if response.output_parsed is None else response.output_parsed.items
    if not raw_items:
        return None
    item = raw_items[0]
    # Model động dùng Enum cho dimension -> đổi về str thuần trước khi trả ra ngoài, để
    # phần còn lại (và Java) chỉ thấy chuỗi như cũ.
    return GeneratedQuizItem(
        dimension_per_statement=[
            str(getattr(value, "value", value))
            for value in item.dimension_per_statement
        ],
        statements=list(item.statements),
        desirability_check=item.desirability_check,
    )


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
