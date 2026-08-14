"""Pricing dùng để tính `cost_usd` báo cáo trong AiUsageRecordedEvent.

Trạng thái từng phần (không còn all-PLACEHOLDER):
- `LLM_PRICING` gpt-5.4/gpt-4o: giá THẬT, đã calibrate từ OpenAI Costs API thật (xem
  spikes/derive_openai_unit_price.py và comment tại chỗ khai báo). gpt-4o-mini/claude-sonnet-4-6:
  vẫn list-price công bố, chưa calibrate (thiếu mẫu / thiếu Admin key tương ứng).
- `LlmUnitPrice.cached_input_per_mtok`: THẬT cho gpt-5.4/gpt-4o (cùng nguồn calibrate ở trên); các
  model còn lại suy từ tỷ lệ discount ~90% các nhà cung cấp công bố cho cache-read, chưa calibrate.
- `DURATION_PRICING_PER_SECOND["azure_stt"]`: giá THẬT, lấy từ Azure Retail Prices API (public,
  không cần key) -- xem comment tại chỗ khai báo bên dưới.
- `DURATION_PRICING_PER_SECOND["azure_tts"]`: vẫn PLACEHOLDER, và chưa từng được dùng (không có
  call site) -- xem comment tại chỗ khai báo.
"""

from typing import Dict, NamedTuple, Optional


class LlmUnitPrice(NamedTuple):
    input_per_mtok: float
    output_per_mtok: float
    # $/1M cache-READ input token (KHÔNG phải cache-write/cache-creation, giá khác hẳn --
    # xem ghi chú ở record_llm_usage). None = model không hỗ trợ cache hoặc chưa biết giá.
    # Số dưới đây suy từ tỷ lệ ~90% discount cả Anthropic lẫn OpenAI đang công bố cho cache
    # read (input_per_mtok * 0.10) -- CHƯA phải số tự calibrate từ Usage/Costs API thật, chỉ
    # là bước sửa tạm để không còn tính cache ở giá full nữa. Thay bằng số derive thật khi có
    # Admin key (spikes/check_claude_cost.py, spikes/derive_openai_unit_price.py).
    cached_input_per_mtok: Optional[float] = None


# Key khớp với tên model truyền vào ChatOpenAI(model=...)/ChatAnthropic(model=...) ở các call site
# (confidence_utils.py, validity_node_config.py, followup_decision_node_config.py) -- bare alias,
# KHÔNG phải dated snapshot ("gpt-5.4-2026-03-05") mà OpenAI Usage/Costs API trả về, xem
# spikes/derive_openai_unit_price.py.
#
# gpt-5.4/gpt-4o: giá THẬT, calibrate từ OPENAI_ADMIN_KEY đã có sẵn trong .env, chạy
#   `uv run python spikes/derive_openai_unit_price.py --model <tên> --days 30` (30 ngày gần nhất,
#   chạy lúc 2026-08-14) -- input/output KHÁC ĐÁNG KỂ so với số PLACEHOLDER cũ (gpt-5.4 input
#   1.00->2.50, output 5.00->15.00; gpt-4o input 1.00->2.4471, output 5.00->10.00). Re-run định kỳ,
#   giá có thể đổi.
# gpt-4o-mini: input THẬT (đủ mẫu, khớp luôn với PLACEHOLDER cũ = 0.15); output/cached_input_per_mtok
#   CHƯA đủ mẫu (<10k token/ngày mọi ngày trong 30 ngày qua) -- giữ PLACEHOLDER cũ cho 2 số này.
# claude-sonnet-4-6: vẫn PLACEHOLDER toàn bộ -- chưa có ANTHROPIC_ADMIN_API_KEY trong .env, xem
#   spikes/check_claude_cost.py.
LLM_PRICING: Dict[str, LlmUnitPrice] = {
    "gpt-5.4": LlmUnitPrice(input_per_mtok=2.50, output_per_mtok=15.00, cached_input_per_mtok=0.25),
    "gpt-4o": LlmUnitPrice(input_per_mtok=2.4471, output_per_mtok=10.00, cached_input_per_mtok=1.25),
    "gpt-4o-mini": LlmUnitPrice(input_per_mtok=0.15, output_per_mtok=0.60, cached_input_per_mtok=0.015),
    "claude-sonnet-4-6": LlmUnitPrice(input_per_mtok=3.00, output_per_mtok=15.00, cached_input_per_mtok=0.30),
}
DEFAULT_LLM_PRICE = LlmUnitPrice(input_per_mtok=1.00, output_per_mtok=5.00, cached_input_per_mtok=0.10)

# $/giây cho dịch vụ tính theo duration.
# azure_stt: giá THẬT, lấy từ Azure Retail Prices API (không cần key, public) -- meter "S1 Speech
# To Text" = $1.00/giờ region southeastasia, quy đổi $/giây. Re-run nếu Azure đổi giá hoặc đổi
# region: uv run python spikes/fetch_azure_retail_prices.py
# azure_tts: KHÔNG dùng field này -- chưa từng có call site nào gọi record_duration_usage cho
# "azure_tts" (Neural TTS rời thật ra tính theo KÍ TỰ, $/1M ký tự, không phải $/giây -- xem
# spikes/fetch_azure_retail_prices.py). Giữ key tồn tại để duration_price_per_second_for() không
# lỗi nếu lỡ có call site gọi nhầm, nhưng PLACEHOLDER 0.006 này sai đơn vị hoàn toàn nếu dùng thật.
DURATION_PRICING_PER_SECOND: Dict[str, float] = {
    "azure_stt": 0.00027778,
    "azure_tts": 0.006,
}
DEFAULT_DURATION_PRICE_PER_SECOND = 0.006


def llm_price_for(model: str) -> LlmUnitPrice:
    return LLM_PRICING.get(model, DEFAULT_LLM_PRICE)


def duration_price_per_second_for(provider: str) -> float:
    return DURATION_PRICING_PER_SECOND.get(provider, DEFAULT_DURATION_PRICE_PER_SECOND)
