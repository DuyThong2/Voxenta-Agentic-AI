"""PLACEHOLDER pricing dùng để tính `cost_usd` báo cáo trong AiUsageRecordedEvent.

Đây là số giả định, KHÔNG phải giá thật từ nhà cung cấp -- cần thay bằng giá thật
($/1M token cho LLM, $/giây cho dịch vụ tính theo duration) trước khi dùng ở
production. Sau khi có đủ dữ liệu ai_usage_record thật ở phía Java, đối chiếu lại
các hằng số này với hoá đơn thật của Anthropic/OpenAI/Azure.
"""

from typing import Dict, NamedTuple


class LlmUnitPrice(NamedTuple):
    input_per_mtok: float
    output_per_mtok: float


# PLACEHOLDER -- khớp key với tên model truyền vào ChatOpenAI(model=...)/ChatAnthropic(model=...)
# ở các call site (confidence_utils.py, validity_node_config.py, followup_decision_node_config.py).
LLM_PRICING: Dict[str, LlmUnitPrice] = {
    "gpt-5.4": LlmUnitPrice(input_per_mtok=1.00, output_per_mtok=5.00),
    "gpt-4o": LlmUnitPrice(input_per_mtok=1.00, output_per_mtok=5.00),
    "claude-sonnet-4-6": LlmUnitPrice(input_per_mtok=3.00, output_per_mtok=15.00),
}
DEFAULT_LLM_PRICE = LlmUnitPrice(input_per_mtok=1.00, output_per_mtok=5.00)

# PLACEHOLDER -- $/giây cho dịch vụ tính theo duration (STT/TTS).
DURATION_PRICING_PER_SECOND: Dict[str, float] = {
    "azure_stt": 0.006,
    "azure_tts": 0.006,
}
DEFAULT_DURATION_PRICE_PER_SECOND = 0.006


def llm_price_for(model: str) -> LlmUnitPrice:
    return LLM_PRICING.get(model, DEFAULT_LLM_PRICE)


def duration_price_per_second_for(provider: str) -> float:
    return DURATION_PRICING_PER_SECOND.get(provider, DEFAULT_DURATION_PRICE_PER_SECOND)
