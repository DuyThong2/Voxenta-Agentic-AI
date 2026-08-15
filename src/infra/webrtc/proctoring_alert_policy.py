"""
Proctoring alert decision policy: when is a YOLO detection a real, reportable
violation vs. single-frame detector noise, and how often should the same
violation type be allowed to re-alert while it keeps holding.

Two gates, applied in order by the caller (controller/proctoring_frame_processor.py):
1. condition_confirmed -- hysteresis: the condition must hold for ALERT_STREAK_FRAMES
   consecutive processed frames before it's treated as real.
2. should_emit_alert -- edge-trigger + cooldown: fires immediately the first time,
   then at most once per ALERT_COOLDOWN_SECONDS while the condition keeps holding.
"""

import asyncio
import logging
import time
from collections import defaultdict
from typing import Dict, Set

from config.webrtc_config import settings
from infra.alert_client import push_alert
# Một chiều, không tạo vòng: proctoring_session cố ý không import ngược lại module này (nó nhận
# on_cleanup từ bên gọi chính vì lý do đó), nên phía quyết định được phép đọc sổ danh tính của nó.
from infra.webrtc import proctoring_session

logger = logging.getLogger(__name__)

ALERT_COOLDOWN_SECONDS = settings.PROCTORING_ALERT_COOLDOWN_SECONDS
ALERT_STREAK_FRAMES = settings.PROCTORING_ALERT_STREAK_FRAMES

# session_id -> set of alert types currently considered "active" (condition still holds)
_active_alert_types: Dict[str, Set[str]] = defaultdict(set)
# session_id -> {alert_type: monotonic time it was last emitted}
_alert_last_emitted_at: Dict[str, Dict[str, float]] = defaultdict(dict)
# session_id -> {alert_type: consecutive processed-frame count the condition has held}
_condition_streak: Dict[str, Dict[str, int]] = defaultdict(dict)

# Vật thể YOLO tìm nhưng KHÔNG đáng cảnh báo. Bàn phím và chuột gần như luôn nằm trong khung: app
# thi chạy trên desktop Windows, thí sinh đang ngồi trước bàn phím. Gộp chúng vào một cảnh báo vật
# thể cấm nghĩa là MỌI phiên thi đều có cảnh báo đỏ -- và một mức CRITICAL bật ở mọi phiên thì giám
# thị học cách bỏ qua nó, tức là mất luôn tác dụng của mức cao nhất với những ca đáng báo thật.
_BENIGN_OBJECTS = {"keyboard", "mouse"}
_PHONE_OBJECTS = {"cell phone"}

# Các loại cảnh báo sinh ra từ vật thể. Bên gọi lặp qua đúng danh sách này để áp hai cổng và để
# xoá streak khi vật thể biến mất -- giữ ở đây để nó nằm cùng chỗ với object_alert_type, thứ duy
# nhất quyết định một nhãn sẽ thành loại nào.
OBJECT_ALERT_TYPES = ("PHONE_DETECTED", "PROHIBITED_OBJECT")


def object_alert_type(label: str) -> str:
    """Nhãn YOLO -> loại cảnh báo, hoặc chuỗi rỗng nếu vật thể đó không đáng báo.

    Mặc định là ĐÁNG BÁO: chỉ những nhãn nằm trong danh sách vô hại mới bị bỏ qua. Detector vốn chỉ
    tìm đúng một danh sách nhãn cố định do người viết chọn, nên một nhãn được thêm vào đó về sau là
    thứ ai đó CỐ Ý muốn thấy -- im lặng bỏ qua nó sẽ là cái im lặng không ai phát hiện ra.
    """
    normalized = str(label or "").strip().lower()
    if not normalized or normalized in _BENIGN_OBJECTS:
        return ""
    if normalized in _PHONE_OBJECTS:
        return "PHONE_DETECTED"
    return "PROHIBITED_OBJECT"


def map_alert_type(event: dict) -> str:
    """Sự kiện SSE -> loại cảnh báo.

    Giờ là ánh xạ đồng nhất: proctoring_frame_processor đã dựng sự kiện bằng đúng từ vựng cảnh báo
    (PHONE_DETECTED/PROHIBITED_OBJECT/PERSON_MISSING/MULTIPLE_PERSONS), nên không còn phép dịch nào
    ở đây nữa. Trước kia chỗ này phải dịch OBJECT_DETECTED sang loại thật, và việc dịch muộn như thế
    là lý do hai cổng hysteresis/cooldown không áp được cho vật thể -- bên gọi lúc đó chưa biết cảnh
    báo sắp mang tên gì để mà gộp.
    """
    return str(event.get("type") or "")


def should_emit_alert(session_id: str, alert_type: str) -> bool:
    """Edge-triggered + cooldown gate: fires immediately the first time a condition
    appears, then at most once per ALERT_COOLDOWN_SECONDS while it keeps holding."""
    now = time.monotonic()
    active = _active_alert_types[session_id]
    last_emitted = _alert_last_emitted_at[session_id]

    is_new = alert_type not in active
    cooldown_elapsed = alert_type not in last_emitted or (now - last_emitted[alert_type]) >= ALERT_COOLDOWN_SECONDS

    if not (is_new or cooldown_elapsed):
        return False

    active.add(alert_type)
    last_emitted[alert_type] = now
    return True


def clear_alert(session_id: str, alert_type: str) -> None:
    """Condition no longer holds -- next occurrence is treated as a fresh edge, not
    a continuation still waiting out the old cooldown."""
    _active_alert_types[session_id].discard(alert_type)


def condition_confirmed(session_id: str, alert_type: str) -> bool:
    """Hysteresis: increments the consecutive-frame streak for this condition and returns
    True only once it has held for ALERT_STREAK_FRAMES processed frames in a row. Call
    reset_streak instead the moment the condition stops holding."""
    streaks = _condition_streak[session_id]
    streaks[alert_type] = streaks.get(alert_type, 0) + 1
    return streaks[alert_type] >= ALERT_STREAK_FRAMES


def reset_streak(session_id: str, alert_type: str) -> None:
    _condition_streak[session_id][alert_type] = 0


def schedule_alert(session_id: str, event: dict) -> None:
    alert_type = map_alert_type(event)
    if not alert_type:
        return

    # Lấy danh tính đã ghi lúc bắt tay WebRTC, thay vì nhân bản session_id ra cả ba trường như trước.
    # Đó là lý do màn hình giám sát in ra UUID chỗ đáng lẽ là tên học viên - và tệ hơn, là lý do cảnh
    # báo AI không gắn được vào ô nào trên lưới. Thiếu thì để rỗng: vox-streaming tra bù được, còn id
    # bịa thì nó không phân biệt nổi với id thật.
    identity = proctoring_session.get_identity(session_id)

    # session_id ở đây là khoá CỤC BỘ của kết nối, và với đường relay nó là uuid4 tự sinh -- không
    # bên nào ngoài process này tra ra nổi. Khoá đối ngoại nằm trong sổ danh tính (register_identity
    # ghi lúc bắt tay). Rơi về khoá cục bộ chỉ khi sổ trống, và đúng cho đường WPF nối thẳng: ở đó
    # khoá cục bộ vốn đã là exam_attempt_id.
    exam_session_id = identity.get("exam_session_id") or session_id

    try:
        asyncio.get_running_loop().create_task(
            push_alert(
                session_id=exam_session_id,
                participant_id=identity.get("participant_id", ""),
                stream_id=identity.get("stream_id", ""),
                alert_type=alert_type,
                confidence=float(event.get("confidence") or 1.0),
                stream_type=identity.get("stream_type", ""),
                detail=str(event.get("message") or ""),
            )
        )
    except RuntimeError:
        logger.warning("[PROCTORING] no running loop, skipping alert for session=%s", session_id)


def clear_session(session_id: str) -> None:
    """Drop all per-session policy state -- called by webrtc_session.cleanup_session
    once a session ends, so a later reused session_id starts with a clean slate."""
    _active_alert_types.pop(session_id, None)
    _alert_last_emitted_at.pop(session_id, None)
    _condition_streak.pop(session_id, None)
