"""Pricing dùng để tính `cost_usd` báo cáo trong AiUsageRecordedEvent.

Trạng thái từng phần (không còn all-PLACEHOLDER):
- `LLM_PRICING` gpt-5.4/gpt-4o/gpt-4o-mini input+output: giá THẬT, calibrate từ OpenAI Costs API
  thật (xem spikes/derive_openai_unit_price.py và comment tại chỗ khai báo).
- `claude-sonnet-4-6`: KHÔNG calibrate được từ spend thật (không tạo được Admin key -- không phải
  Owner/không có org) -- nhưng đã đối chiếu khớp với trang giá chính thức của Anthropic (xem comment
  tại chỗ khai báo). Đáng tin hơn hẳn placeholder tự đoán, nhưng vẫn là giá NIÊM YẾT chứ không phải
  giá đã thật sự chi tiêu.
- `LlmUnitPrice.cached_input_per_mtok`: THẬT cho gpt-5.4/gpt-4o (cùng nguồn calibrate ở trên);
  gpt-4o-mini/claude-sonnet-4-6 suy từ tỷ lệ discount ~90% các nhà cung cấp công bố cho cache-read
  (0 dữ liệu cache thật cho gpt-4o-mini, không tạo được Admin key cho claude) -- xem comment.
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
    # $/1M cache-WRITE (cache_creation) input token -- KHÁC cache-READ ở trên, giá thường ĐẮT
    # hơn input thường chứ không rẻ hơn. None = fallback về input_per_mtok (đúng hành vi thật
    # cho OpenAI: viết cache không tính phí thêm, chỉ đọc cache mới được giảm giá -- xem comment
    # tại chỗ khai báo LLM_PRICING). Verify 2026-08-15: cache_creation_input_tokens = 0 trên MỌI
    # row ai_usage_record thật tính đến nay (cả OpenAI lẫn Anthropic) -- field này hiện KHÔNG ảnh
    # hưởng cost_usd thật nào, chỉ là phòng trước cho lúc có ai bật cache_control cho Claude.
    cache_creation_per_mtok: Optional[float] = None


# Key khớp với tên model truyền vào ChatOpenAI(model=...)/ChatAnthropic(model=...) ở các call site
# (confidence_utils.py, validity_node_config.py, followup_decision_node_config.py) -- bare alias,
# KHÔNG phải dated snapshot ("gpt-5.4-2026-03-05") mà OpenAI Usage/Costs API trả về, xem
# spikes/derive_openai_unit_price.py.
#
# gpt-5.4/gpt-4o/gpt-4o-mini input+output: giá THẬT, calibrate từ OPENAI_ADMIN_KEY đã có sẵn trong
#   .env, chạy `uv run python spikes/derive_openai_unit_price.py --model <tên> --days <N>` (chạy
#   lúc 2026-08-14, gpt-5.4/gpt-4o dùng 30 ngày, gpt-4o-mini dùng 90 ngày -- traffic thấp hơn nên
#   cần cửa sổ rộng hơn để đủ mẫu, xem MIN_TOKENS_TOTAL trong script). gpt-4o-mini output khớp
#   chính xác PLACEHOLDER cũ ($0.60) -- input cũng khớp ($0.15), chỉ gpt-5.4/gpt-4o lệch đáng kể so
#   với PLACEHOLDER cũ (gpt-5.4 input 1.00->2.50, output 5.00->15.00; gpt-4o input 1.00->2.4472,
#   output 5.00->9.3945). Re-run định kỳ, giá có thể đổi.
# gpt-4o-mini cached_input_per_mtok: KHÔNG derive được -- 0 token cache thật nào ghi nhận trong 90
#   ngày qua (không phải do thiếu mẫu, mà do model này chưa từng thật sự dùng prompt caching trong
#   traffic hiện tại). Giữ nguyên số suy theo tỷ lệ chuẩn (input*0.10) làm nền, phòng khi phát sinh.
# claude-sonnet-4-6: KHÔNG calibrate được từ spend thật -- không tạo được Admin key (không phải
#   Owner/không có org, xem spikes/check_claude_cost.py để dùng sau nếu có key). Đã đối chiếu số
#   dưới đây khớp CHÍNH XÁC với trang giá chính thức platform.claude.com/docs/en/about-claude/pricing
#   (fetch 2026-08-14: Sonnet 4.6 = $3 input / $15 output / $0.30 cache-hit = input*0.1) -- là giá
#   niêm yết Anthropic đang áp dụng, KHÔNG phải giá đã thật sự chi tiêu, nhưng đáng tin hơn hẳn số
#   tự đoán trước đó. Trang này không có JSON API ổn định để tự động hoá (khác Azure Retail Prices)
#   -- re-check thủ công định kỳ nếu Anthropic đổi giá.
# cache_creation_per_mtok: OpenAI (gpt-5.4/gpt-4o/gpt-4o-mini) để None -- OpenAI KHÔNG tính phí
#   thêm khi ghi cache (platform.openai.com/docs/guides/prompt-caching: "no additional cost for
#   writing to the cache"), nên fallback về input_per_mtok là đúng giá thật, không phải placeholder.
#   claude-sonnet-4-6 = input*1.25 = $3.75 -- Anthropic tính cache-write đắt hơn input thường (1.25x
#   cho TTL ephemeral mặc định 5 phút, 2x cho TTL mở rộng 1 giờ). Repo này KHÔNG dùng TTL 1 giờ
#   (không có config `cache_control`/`ttl` nào trong code tại thời điểm viết, xác nhận bằng grep) --
#   giữ nguyên 1.25x cho tới khi có chỗ nào bật extended cache. Verify 2026-08-15: field này = 0 ở
#   MỌI row ai_usage_record thật hiện có (không nơi nào trong repo từng gọi cache_control cho
#   Claude) -- số 3.75 hiện chưa ảnh hưởng cost_usd thật nào, chỉ là phòng trước.
LLM_PRICING: Dict[str, LlmUnitPrice] = {
    "gpt-5.4": LlmUnitPrice(input_per_mtok=2.50, output_per_mtok=15.00, cached_input_per_mtok=0.25),
    "gpt-4o": LlmUnitPrice(input_per_mtok=2.4472, output_per_mtok=9.3945, cached_input_per_mtok=1.25),
    "gpt-4o-mini": LlmUnitPrice(input_per_mtok=0.15, output_per_mtok=0.60, cached_input_per_mtok=0.015),
    "claude-sonnet-4-6": LlmUnitPrice(
        input_per_mtok=3.00, output_per_mtok=15.00, cached_input_per_mtok=0.30, cache_creation_per_mtok=3.75
    ),
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
# azure_voice_live_input: ƯỚC LƯỢNG (không phải giá Azure thực báo cáo) -- Voice Live
# (infra/voice_live_client.py) chỉ dùng cho STT+VAD (turn_detection.create_response=False), nên
# SDK không bao giờ bắn event response.done mang usage/token thật (event đó chỉ fire khi model
# tự tạo response). Suy từ: (a) tỷ lệ OpenAI công bố cho model nền -- 1 audio input token / 100ms
# = 10 token/giây (developers.openai.com/api/docs/guides/realtime-costs, fetch 2026-08-14) -- CHƯA
# xác nhận độc lập là đúng y hệt cho Azure Voice Live; (b) giá thật "Voice Live API Std - Standard
# Speech Audio Input Tokens" = $0.015/1K token (Azure Retail Prices API, region southeastasia,
# xác nhận 2026-08-14). => 10 * 0.015 / 1000 = $0.00015/giây. Tính trên TOÀN BỘ byte audio đã push
# qua push_audio() kể cả khoảng lặng (app không gate VAD phía client trước khi gửi) -- có thể lệch
# so với cách Azure thực sự tokenize im lặng. Xem VoiceLiveClient.pop_input_audio_duration_ms().
DURATION_PRICING_PER_SECOND: Dict[str, float] = {
    "azure_stt": 0.00027778,
    "azure_tts": 0.006,
    "azure_voice_live_input": 0.00015,
}
DEFAULT_DURATION_PRICE_PER_SECOND = 0.006


def llm_price_for(model: str) -> LlmUnitPrice:
    return LLM_PRICING.get(model, DEFAULT_LLM_PRICE)


def duration_price_per_second_for(provider: str) -> float:
    return DURATION_PRICING_PER_SECOND.get(provider, DEFAULT_DURATION_PRICE_PER_SECOND)
