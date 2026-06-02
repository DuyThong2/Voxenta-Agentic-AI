"""
WebRTC signaling + YOLO proctoring controller.

Supports up to MAX_CONNECTIONS concurrent peer connections.
Each connection has its own event log accessible via SSE or REST.
"""

import asyncio
import json
import logging
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from ultralytics import YOLO

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webrtc", tags=["WebRTC"])

# --------------- Configuration ---------------
MAX_CONNECTIONS = int(os.getenv("WEBRTC_MAX_CONNECTIONS", "100"))
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
FRAME_SKIP = int(os.getenv("YOLO_FRAME_SKIP", "10"))
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.5"))

# --------------- State ---------------
pcs: Set[RTCPeerConnection] = set()
# session_id -> RTCPeerConnection
session_map: Dict[str, RTCPeerConnection] = {}
# session_id -> list of events
session_events: Dict[str, List[dict]] = defaultdict(list)
# session_id -> set of SSE subscribers (asyncio.Queue)
session_subscribers: Dict[str, Set[asyncio.Queue]] = defaultdict(set)

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


async def broadcast_event(session_id: str, event: dict):
    """Push event to all SSE subscribers of a session."""
    session_events[session_id].append(event)
    for queue in session_subscribers.get(session_id, set()):
        await queue.put(event)


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

        # Save first frame as debug image to verify video is not black/corrupt
        if frame_count == FRAME_SKIP:
            debug_dir = Path("debug_frames")
            debug_dir.mkdir(exist_ok=True)
            debug_path = debug_dir / f"frame_{session_id[:8]}.png"
            try:
                import cv2
                cv2.imwrite(str(debug_path), img)
                logger.info("[YOLO_DEBUG] Saved debug frame to %s", debug_path)
            except Exception as exc:
                logger.warning("[YOLO_DEBUG] Failed to save frame: %s", exc)

        # Log frame info every 50 processed frames for debugging
        if frame_count % (FRAME_SKIP * 5) == 0:
            h, w = img.shape[:2]
            logger.info(
                "[YOLO_DEBUG] session=%s frame=%d size=%dx%d dtype=%s",
                session_id, frame_count, w, h, img.dtype,
            )

        events = []
        person_count = 0

        try:
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

                if label in ("cell phone", "book", "laptop", "keyboard", "mouse"):
                    events.append(
                        build_event(
                            event_type="OBJECT_DETECTED",
                            object=label,
                            confidence=confidence,
                            message=f"Phát hiện vật thể nghi vấn: {label}",
                        )
                    )

        if frame_count % (FRAME_SKIP * 5) == 0:
            logger.info(
                "[YOLO_DEBUG] session=%s detections=%s person_count=%d",
                session_id, raw_detections or "none", person_count,
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
            await broadcast_event(session_id, event)
            logger.info("[PROCTORING] session=%s %s", session_id, json.dumps(event, ensure_ascii=False))


async def cleanup_session(session_id: str):
    """Clean up a session's peer connection and resources."""
    pc = session_map.pop(session_id, None)
    if pc:
        await pc.close()
        pcs.discard(pc)

    # Notify SSE subscribers that session ended
    for queue in session_subscribers.get(session_id, set()):
        await queue.put({"type": "SESSION_ENDED", "timestamp": utc_now()})
    session_subscribers.pop(session_id, None)

    logger.info("[WEBRTC] Session %s cleaned up. Active: %d", session_id, len(pcs))


# --------------- REST Endpoints ---------------

@router.post("/offer")
async def offer(request: Request):
    """
    WebRTC signaling: browser sends SDP offer, server returns SDP answer.
    Creates a new proctoring session with YOLO video analysis.
    """
    if len(pcs) >= MAX_CONNECTIONS:
        return JSONResponse(
            status_code=503,
            content={
                "error": f"Max connections ({MAX_CONNECTIONS}) reached. Try again later.",
                "active_connections": len(pcs),
            },
        )

    params = await request.json()

    if "sdp" not in params or "type" not in params:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid WebRTC offer. Body must include 'sdp' and 'type'."},
        )

    session_id = str(uuid.uuid4())
    pc = RTCPeerConnection()
    pcs.add(pc)
    session_map[session_id] = pc

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("[WEBRTC] Session %s state: %s", session_id, pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await cleanup_session(session_id)

    @pc.on("track")
    def on_track(track):
        logger.info("[WEBRTC] Track received: %s for session %s", track.kind, session_id)
        if track.kind == "video":
            asyncio.create_task(process_video_track(track, session_id))

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    )

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse(
        {
            "session_id": session_id,
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }
    )


@router.get("/connections")
def active_connections():
    """Return count and list of active sessions."""
    return {
        "active_connections": len(pcs),
        "max_connections": MAX_CONNECTIONS,
        "sessions": list(session_map.keys()),
    }


@router.get("/connections/{session_id}/events")
def get_session_events(session_id: str, limit: int = 100):
    """Get stored proctoring events for a session."""
    if session_id not in session_map and session_id not in session_events:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    events = session_events.get(session_id, [])
    return {
        "session_id": session_id,
        "total": len(events),
        "events": events[-limit:],
    }


@router.get("/connections/{session_id}/events/stream")
async def stream_session_events(session_id: str):
    """
    SSE endpoint: stream proctoring events in real-time.
    Client should use EventSource to connect.
    """
    if session_id not in session_map and session_id not in session_events:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    queue: asyncio.Queue = asyncio.Queue()
    session_subscribers[session_id].add(queue)

    async def event_generator():
        try:
            # Send existing events first
            for event in session_events.get(session_id, []):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            # Stream new events
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    if event.get("type") == "SESSION_ENDED":
                        break
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f": keepalive {utc_now()}\n\n"
        finally:
            session_subscribers[session_id].discard(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/connections/{session_id}")
async def disconnect_session(session_id: str):
    """Force disconnect a session."""
    if session_id not in session_map:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    await cleanup_session(session_id)
    return {"status": "disconnected", "session_id": session_id}


# --------------- Lifecycle helpers ---------------

async def close_all_connections():
    """Called on app shutdown to clean up all peer connections."""
    logger.info("[WEBRTC] Closing %d active connections...", len(pcs))
    for session_id in list(session_map.keys()):
        await cleanup_session(session_id)
    logger.info("[WEBRTC] All connections closed.")
