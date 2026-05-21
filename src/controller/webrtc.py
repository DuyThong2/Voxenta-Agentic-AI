import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Set

from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from ultralytics import YOLO

router = APIRouter(prefix="/webrtc", tags=["WebRTC"])

pcs: Set[RTCPeerConnection] = set()

YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
yolo_model = YOLO(YOLO_MODEL)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_event(event_type: str, message: str, confidence: float = 0.9, **extra):
    return {
        "type": event_type,
        "confidence": confidence,
        "message": message,
        "timestamp": utc_now(),
        **extra,
    }


async def process_video_track(track):
    """
    Nhận livestream video frame-by-frame từ WebRTC.
    Bản prototype Python 3.12: dùng YOLO để detect person/object.
    """

    frame_count = 0

    while True:
        try:
            frame = await track.recv()
        except Exception as exc:
            print("[WEBRTC] Stop receiving video track:", repr(exc))
            break

        frame_count += 1

        # Giảm tải CPU: chỉ xử lý mỗi 10 frame
        if frame_count % 10 != 0:
            continue

        img = frame.to_ndarray(format="bgr24")

        events = []
        person_count = 0

        try:
            yolo_results = yolo_model(img, verbose=False)
        except Exception as exc:
            print("[YOLO_ERROR]", repr(exc))
            continue

        for result in yolo_results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = yolo_model.names[cls_id]

                if confidence < 0.5:
                    continue

                if label == "person":
                    person_count += 1

                if label in ["cell phone", "book", "laptop", "keyboard", "mouse"]:
                    events.append(
                        build_event(
                            event_type="OBJECT_DETECTED",
                            object=label,
                            confidence=confidence,
                            message=f"Phát hiện vật thể nghi vấn: {label}",
                        )
                    )

        if person_count == 0:
            events.append(
                build_event(
                    event_type="PERSON_MISSING",
                    message="Không thấy người trong camera",
                    confidence=0.9,
                )
            )

        if person_count > 1:
            events.append(
                build_event(
                    event_type="MULTIPLE_PERSONS",
                    message="Phát hiện nhiều hơn một người trong camera",
                    confidence=0.9,
                    person_count=person_count,
                )
            )

        for event in events:
            print("[PROCTORING_EVENT]", json.dumps(event, ensure_ascii=False))


@router.post("/offer")
async def offer(request: Request):
    """
    React gửi WebRTC offer vào endpoint này.
    Python trả answer để browser bắt đầu livestream video.
    """

    params = await request.json()

    if "sdp" not in params or "type" not in params:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid WebRTC offer. Body must include 'sdp' and 'type'."
            },
        )

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("[WEBRTC] Connection state:", pc.connectionState)

        if pc.connectionState in ["failed", "closed", "disconnected"]:
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        print("[WEBRTC] Track received:", track.kind)

        if track.kind == "video":
            asyncio.create_task(process_video_track(track))

    await pc.setRemoteDescription(
        RTCSessionDescription(
            sdp=params["sdp"],
            type=params["type"],
        )
    )

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse(
        {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }
    )


@router.get("/connections")
def active_connections():
    return {
        "active_connections": len(pcs)
    }