"""
Proctoring video frame processing: receive WebRTC video frames, run YOLO
detection, and turn raw detections into proctoring events via
proctoring_alert_policy (hysteresis/cooldown) before storing/broadcasting them.
"""

import json
import logging

from ultralytics import YOLO

from config.webrtc_config import settings
from controller import gpu_scheduler
from infra.webrtc import proctoring_alert_policy
from infra.webrtc import proctoring_session as webrtc_session

logger = logging.getLogger(__name__)

YOLO_MODEL = settings.YOLO_MODEL
FRAME_SKIP = settings.YOLO_FRAME_SKIP
YOLO_CONFIDENCE = settings.YOLO_CONFIDENCE

yolo_model = YOLO(YOLO_MODEL)

# YOLO nhận ra 80 lớp COCO; đây là những lớp buổi thi thật sự quan tâm. Danh sách này CỐ Ý hẹp:
# ghế, chai nước, màn hình... đều có mặt hợp lệ trên bàn thi, và báo về chúng chỉ làm ngập lưới giám
# sát. proctoring_alert_policy.object_alert_type quyết định trong số này cái nào thành cảnh báo gì.
_WATCHED_OBJECTS = ("cell phone", "book", "laptop", "keyboard", "mouse")


async def process_video_track(track, session_id: str):
    """
    Receive video frames from WebRTC, run YOLO detection,
    store and broadcast proctoring events.
    """
    frame_count = 0

    while True:
        try:
            frame = await track.recv()
        except Exception as exc:
            logger.info("[WEBRTC] Track ended for session %s: %s", session_id, exc)
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        img = frame.to_ndarray(format="bgr24")

        # Debug: check if frame is all black (all zeros)
        if frame_count <= FRAME_SKIP * 3:
            import numpy as np
            mean_val = np.mean(img)
            max_val = np.max(img)
            non_zero = np.count_nonzero(img)
            total = img.size
            logger.info(
                "[FRAME_DEBUG] session=%s frame=%d mean=%.2f max=%d non_zero=%d/%d (%.1f%%) "
                "pts=%s time_base=%s format=%s size=%s",
                session_id, frame_count, mean_val, max_val, non_zero, total,
                100 * non_zero / total,
                frame.pts, frame.time_base, frame.format.name, (frame.width, frame.height),
            )

        # Log frame info every 50 processed frames for debugging
        if frame_count % (FRAME_SKIP * 5) == 0:
            h, w = img.shape[:2]
            logger.info(
                "[YOLO_DEBUG] session=%s frame=%d size=%dx%d dtype=%s",
                session_id, frame_count, w, h, img.dtype,
            )

        events = []
        person_count = 0
        # alert_type -> (nhãn, độ tin cậy) của vật thể tiêu biểu nhất cho loại đó trong khung này.
        # Gom theo LOẠI chứ không theo nhãn: sách và laptop cùng là PROHIBITED_OBJECT, và giám thị
        # cần biết "có tài liệu cấm trong khung", không cần hai dòng riêng cho hai vật.
        object_candidates: dict[str, tuple[str, float]] = {}

        try:
            # Shares the GPU with realtime/avatar_renderer.py's LivePortrait/MuseTalk inference --
            # see gpu_scheduler.py for why this needs to take turns rather than run concurrently.
            async with gpu_scheduler.gpu_lock():
                yolo_results = yolo_model(img, verbose=False)
        except Exception as exc:
            logger.warning("[YOLO_ERROR] session=%s: %s", session_id, exc)
            continue

        # Log all raw detections periodically
        raw_detections = []
        for result in yolo_results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = yolo_model.names[cls_id]
                raw_detections.append(f"{label}({confidence:.2f})")

                if confidence < YOLO_CONFIDENCE:
                    continue

                if label == "person":
                    person_count += 1
                    continue

                if label not in _WATCHED_OBJECTS:
                    continue

                alert_type = proctoring_alert_policy.object_alert_type(label)
                if not alert_type:
                    continue

                current = object_candidates.get(alert_type)
                if current is None or confidence > current[1]:
                    object_candidates[alert_type] = (label, confidence)

        if frame_count % (FRAME_SKIP * 5) == 0:
            logger.info(
                "[YOLO_DEBUG] session=%s detections=%s person_count=%d",
                session_id, raw_detections or "none", person_count,
            )

        # Vật thể đi qua ĐÚNG hai cổng như hai điều kiện về người ở dưới. Trước đây chúng không qua
        # cổng nào: mỗi khung hình có điện thoại đẻ ra một sự kiện, nên một chiếc điện thoại nằm yên
        # trên bàn 30 giây thành hàng chục cảnh báo giống hệt nhau -- vừa làm ngập sổ bằng chứng,
        # vừa dạy giám thị bỏ qua đúng loại cảnh báo nặng nhất. Hysteresis còn loại được nhấp nháy
        # một khung của detector, thứ chiếm phần lớn báo động giả.
        for object_alert_type in proctoring_alert_policy.OBJECT_ALERT_TYPES:
            candidate = object_candidates.get(object_alert_type)
            if candidate is None:
                proctoring_alert_policy.reset_streak(session_id, object_alert_type)
                proctoring_alert_policy.clear_alert(session_id, object_alert_type)
                continue

            object_label, object_confidence = candidate
            if (
                proctoring_alert_policy.condition_confirmed(session_id, object_alert_type)
                and proctoring_alert_policy.should_emit_alert(session_id, object_alert_type)
            ):
                events.append(
                    webrtc_session.build_event(
                        event_type=object_alert_type,
                        object=object_label,
                        confidence=object_confidence,
                        message=f"Phát hiện vật thể nghi vấn: {object_label}",
                    )
                )

        if person_count == 0:
            if (
                proctoring_alert_policy.condition_confirmed(session_id, "PERSON_MISSING")
                and proctoring_alert_policy.should_emit_alert(session_id, "PERSON_MISSING")
            ):
                events.append(
                    webrtc_session.build_event(
                        event_type="PERSON_MISSING",
                        message="Không thấy người trong camera",
                        confidence=0.9,
                    )
                )
        else:
            proctoring_alert_policy.reset_streak(session_id, "PERSON_MISSING")
            proctoring_alert_policy.clear_alert(session_id, "PERSON_MISSING")

        if person_count > 1:
            if (
                proctoring_alert_policy.condition_confirmed(session_id, "MULTIPLE_PERSONS")
                and proctoring_alert_policy.should_emit_alert(session_id, "MULTIPLE_PERSONS")
            ):
                events.append(
                    webrtc_session.build_event(
                        event_type="MULTIPLE_PERSONS",
                        message="Phát hiện nhiều hơn một người trong camera",
                        confidence=0.9,
                        person_count=person_count,
                    )
                )
        else:
            proctoring_alert_policy.reset_streak(session_id, "MULTIPLE_PERSONS")
            proctoring_alert_policy.clear_alert(session_id, "MULTIPLE_PERSONS")

        for event in events:
            await webrtc_session.broadcast_event(session_id, event)
            logger.info("[PROCTORING] session=%s %s", session_id, json.dumps(event, ensure_ascii=False))
            proctoring_alert_policy.schedule_alert(session_id, event)
