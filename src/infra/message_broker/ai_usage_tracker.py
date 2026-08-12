"""In-memory buffer gom các AiUsageEventItem phát sinh trong lúc xử lý một turn (nhiều
lệnh gọi LLM/STT/TTS có thể xảy ra cho cùng một turn), để turn_publisher.py xả +
publish một lần cùng lúc với AnswerTurnsRecordedEvent của turn đó.

Khoá theo answer_id (không phải answer_id+turn_order): tại thời điểm ghi usage, các
call site (LLM eval, STT, TTS) không phải lúc nào cũng biết chắc turn_order hiện tại
đang thuộc về turn nào sẽ được publish tiếp theo, nên gom theo answer_id rồi xả sạch
buffer của answer_id đó ngay khi publish_turn_if_new chạy xong là đơn giản và đủ
chính xác cho mục đích audit chi phí.

KHÔNG durable (khác published_turn_orders trong turn_publisher.py, vốn đi qua Postgres
checkpoint) -- nếu process restart giữa lúc ghi usage và lúc publish, các usage event
đang buffer sẽ mất. Chấp nhận được vì đây là dữ liệu audit/COGS (ước tính chi phí),
không phải nguồn trừ quota tuyệt đối; cân nhắc làm durable nếu sau này cần độ chính
xác cao hơn.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

from config import ai_usage_pricing as pricing
from events import AiUsageEventItem, AiUsageTokens

logger = logging.getLogger(__name__)

_usage_buffer: Dict[str, List[AiUsageEventItem]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_llm_usage(answer_id: Optional[str], provider: str, model: str, response) -> None:
    """`response` là AIMessage trả về từ langchain `llm.invoke(...)`. Bỏ qua im lặng nếu
    response không mang usage_metadata (một số provider/lỗi gọi không trả về usage)."""
    if not answer_id:
        return

    usage_metadata = getattr(response, "usage_metadata", None)
    if not usage_metadata:
        logger.debug(
            "[ai_usage_tracker] response has no usage_metadata, skip: provider=%s model=%s",
            provider, model,
        )
        return

    input_tokens = usage_metadata.get("input_tokens") or 0
    output_tokens = usage_metadata.get("output_tokens") or 0
    input_token_details = usage_metadata.get("input_token_details") or {}
    price = pricing.llm_price_for(model)
    cost_usd = (
        (input_tokens / 1_000_000) * price.input_per_mtok
        + (output_tokens / 1_000_000) * price.output_per_mtok
    )

    item = AiUsageEventItem(
        usage_event_id=str(uuid4()),
        type="LLM_TOKEN",
        provider=provider,
        model=model,
        usage=AiUsageTokens(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=input_token_details.get("cache_creation"),
            cache_read_input_tokens=input_token_details.get("cache_read"),
        ),
        unit_price={
            "inputPerMtok": price.input_per_mtok,
            "outputPerMtok": price.output_per_mtok,
            "currency": "USD",
        },
        cost_usd=round(cost_usd, 8),
        occurred_at=_now_iso(),
    )
    _usage_buffer.setdefault(answer_id, []).append(item)


def record_duration_usage(answer_id: Optional[str], provider: str, duration_ms: Optional[float]) -> None:
    if not answer_id or duration_ms is None:
        return

    price_per_second = pricing.duration_price_per_second_for(provider)
    cost_usd = (duration_ms / 1000.0) * price_per_second

    item = AiUsageEventItem(
        usage_event_id=str(uuid4()),
        type="DURATION",
        provider=provider,
        duration_ms=int(duration_ms),
        unit_price={"amount": price_per_second, "unit": "per_second", "currency": "USD"},
        cost_usd=round(cost_usd, 8),
        occurred_at=_now_iso(),
    )
    _usage_buffer.setdefault(answer_id, []).append(item)


def pop_usage(answer_id: str) -> List[AiUsageEventItem]:
    """Lấy và xoá toàn bộ usage event đang buffer cho answer_id này."""
    return _usage_buffer.pop(answer_id, [])
