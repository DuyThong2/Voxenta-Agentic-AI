from typing import Dict, List, Literal, Optional

from events.envelope import EventEnvelope
from schemas.common import _CamelMessage


class AiUsageTokens(_CamelMessage):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None


class AiUsageEventItem(_CamelMessage):
    """Một lần gọi AI có chi phí (LLM hoặc dịch vụ tính theo duration như STT/TTS).

    `usage_event_id` là khoá idempotency phía Java (uk_ai_usage_record_usage_event_id) --
    PHẢI unique cho mỗi item, kể cả khi message này được publish lại (retry).
    `type` phải đúng "LLM_TOKEN" hoặc "DURATION" (khớp enum AiUsageType bên Java), không phải
    "llm_usage"/"duration_usage".
    """

    usage_event_id: str
    type: Literal["LLM_TOKEN", "DURATION"]
    provider: str
    model: Optional[str] = None
    usage: Optional[AiUsageTokens] = None
    duration_ms: Optional[int] = None
    unit_price: Dict = {}
    cost_usd: float
    occurred_at: str


class AiUsageRecordedEvent(EventEnvelope):
    event_type: Literal["AiUsageRecorded"] = "AiUsageRecorded"

    exam_session_id: str
    turn_id: str
    usage_events: List[AiUsageEventItem]