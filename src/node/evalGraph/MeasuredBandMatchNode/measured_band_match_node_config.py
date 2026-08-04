"""Khớp bậc cho các tiêu chí do MÁY chấm (phát âm, độ trôi chảy).

Vì sao cần node này -- ba lỗi im lặng chồng nhau, đo được trên dữ liệu thật:

1. `matched_band_code` chưa từng được ghi ở đường Azure. AzureScoreScaleNode chỉ gán
   `criterion.note` qua build_framework_note; trường mã bậc giữ nguyên mặc định rỗng.

2. Không có phép ánh xạ điểm -> bậc nào trong dữ liệu. `framework_result_bands` không có cột
   khoảng điểm; `FrameworkBand.score_min/score_max` bên Java được điền bằng min/max của RUBRIC
   (0-100) cho MỌI bậc, nên `find_band_for_score` so `0 <= score <= 100` với mọi bậc và luôn
   trả về bậc đầu tiên trong thứ tự sắp xếp.

3. Chuỗi rỗng KHÔNG PHẢI NULL, nên nó lọt qua chốt `matched_band_code IS NOT NULL` trong
   findEstimatedResultBandOrder rồi mới chết ở phép nối `band.code = matched_band_code`.

Hệ quả đo được: 5 dòng điểm tiêu chí, chỉ 3 dòng qua được phép nối. Ngưỡng của phép ước lượng
là `total >= 5` đếm SAU khi nối, nên nó trả rỗng -- ước lượng bậc chưa từng chạy, và độ khó câu
hỏi đang lấy theo chính sách trường chứ không theo học sinh. Phát âm và độ trôi chảy vĩnh viễn
không được bỏ phiếu, kể cả sau hàng trăm phiên.

Cách chữa KHÔNG phải bịa một ngưỡng điểm: điểm Azure và bậc VSTEP không cùng thang, đặt cut-off
nào cũng là dựng ra một phép hiệu chỉnh không tồn tại. Bản mô tả bậc theo từng tiêu chí thì có
thật (framework_criterion_bands.descriptor) và viết đúng bằng thứ Azure đo -- "tốc độ chậm và
có nhiều khoảng dừng", "tương đối hiểu được dù ảnh hưởng tiếng mẹ đẻ". Nên hỏi LLM khớp bằng
chứng đo được vào mô tả, cùng cách ba tiêu chí kia đang làm.
"""

import logging
from typing import Any, Dict, List, Optional

from node.evalGraph.MeasuredBandMatchNode.measured_band_match_prompt import (
    build_measured_band_match_prompt,
)
from utils.confidence_utils import call_llm_claude, call_llm_openai, call_with_retry_and_fallback

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You place measured speaking performance onto a band ladder using the band descriptors. "
    "You return only JSON."
)

# Đúng hai tiêu chí này được máy chấm; ba tiêu chí còn lại đã có mã bậc từ
# LanguageQualityEvalNode và không được đụng vào ở đây.
MEASURED_CRITERION_KEYS = ("pronunciation", "fluency")


def _framework_for(criteria_frameworks: List[Any], criterion_key: str) -> Optional[Any]:
    return next(
        (cf for cf in (criteria_frameworks or []) if cf.criterion_key == criterion_key),
        None,
    )


def _metrics_for(criterion: Any) -> Dict[str, Any]:
    """Bằng chứng gửi cho model: điểm đã quy về thang rubric + các thành phần Azure trả về.

    Giữ cả `raw_azure_score` (AzureScoreScaleNode nhét vào subscores) vì mô tả bậc nói về hành
    vi quan sát được, mà thành phần thô mới phản ánh hành vi -- điểm đã quy đổi thì đã trộn
    thang rubric của trường vào.
    """
    metrics: Dict[str, Any] = {"rubric_score": criterion.score}
    for key, value in (criterion.subscores or {}).items():
        if isinstance(value, (int, float)):
            metrics[key] = value
    return metrics


def measured_band_match_node(state: Dict[str, Any]) -> Dict[str, Any]:
    speaking_input = state.get("speaking_input")
    pronunciation_result = state.get("pronunciation_result")
    if speaking_input is None or pronunciation_result is None:
        return {}

    frameworks = speaking_input.criteria_frameworks or []
    entries = []
    for criterion_key in MEASURED_CRITERION_KEYS:
        criterion = getattr(pronunciation_result.criteria, criterion_key, None)
        framework = _framework_for(frameworks, criterion_key)
        # Không có thang bậc thì không có gì để khớp vào; không có điểm thì không có bằng
        # chứng. Cả hai trường hợp đều để rỗng -- rỗng nghĩa là "không biết", và cả chuỗi
        # phía sau xử lý "không biết" đúng cách bằng cách bỏ dòng đó ra khỏi phép bỏ phiếu.
        if criterion is None or framework is None or not framework.bands:
            continue
        if criterion.score is None:
            continue
        entries.append({
            "criterion_key": criterion_key,
            "metrics": _metrics_for(criterion),
            "bands": framework.bands,
        })

    if not entries:
        return {}

    prompt = build_measured_band_match_prompt(entries)
    try:
        payload = call_with_retry_and_fallback(
            lambda: call_llm_openai(_SYSTEM_PROMPT, prompt),
            lambda: call_llm_claude(_SYSTEM_PROMPT, prompt),
        )
    except Exception:
        # Hỏng thì để rỗng, KHÔNG đoán. Một mã bịa ra sẽ được đếm như phiếu thật trong
        # findEstimatedResultBandOrder rồi đi thẳng vào việc chọn độ khó câu hỏi.
        logger.exception("[measured_band_match] không khớp được bậc cho tiêu chí đo bằng số")
        return {}

    matches = (payload or {}).get("matches") or {}
    applied: Dict[str, str] = {}
    for entry in entries:
        criterion_key = entry["criterion_key"]
        raw = (matches.get(criterion_key) or {}).get("band_code")
        code = (raw or "").strip()
        if not code:
            continue
        # Chặn MÃ LẠC ngay tại đây thay vì để nó rơi vào DB. Đây đúng là chế độ hỏng thứ hai
        # mà task/utils/check-matched-band-health.sql cảnh báo: mã không khớp thang thì phép
        # nối rỗng và bản ghi biến mất, không lỗi, không log, chỉ là mẫu nhỏ đi.
        valid = {band.code for band in entry["bands"]}
        if code not in valid:
            logger.warning(
                "[measured_band_match] bỏ mã lạc %r cho %s (thang hợp lệ: %s)",
                code, criterion_key, sorted(valid),
            )
            continue
        getattr(pronunciation_result.criteria, criterion_key).matched_band_code = code
        applied[criterion_key] = code

    return {
        "pronunciation_result": pronunciation_result,
        "metadata": {"measured_band_match": applied},
    }
