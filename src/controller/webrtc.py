"""
WebRTC signaling + YOLO proctoring controller.

Supports up to MAX_CONNECTIONS concurrent peer connections.
Each connection has its own event log accessible via SSE or REST.

Session bookkeeping, the alert decision policy, and the YOLO/aiortc
frame-processing loop all live under infra/webrtc/ (proctoring_session.py,
proctoring_alert_policy.py, proctoring_frame_processor.py, respectively) --
grouped there by feature at explicit user direction, even though the alert
policy is decision logic rather than pure infra relay; this file is just the
thin FastAPI route surface over all three, and the one place that wires
cleanup_session's on_cleanup hook to the policy module's clear_session (so
proctoring_session.py itself never has to import proctoring_alert_policy).
"""

import asyncio
import json
import logging
import uuid

from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config.webrtc_config import build_ice_servers, settings
from infra.webrtc import proctoring_alert_policy
from infra.webrtc import proctoring_session
from infra.webrtc.proctoring_frame_processor import process_video_track

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webrtc", tags=["WebRTC"])

MAX_CONNECTIONS = settings.WEBRTC_MAX_CONNECTIONS


async def _cleanup_session(session_id: str) -> None:
    await proctoring_session.cleanup_session(session_id, on_cleanup=proctoring_alert_policy.clear_session)


async def close_all_connections():
    """Called on app shutdown to clean up all peer connections."""
    await proctoring_session.close_all_connections(on_cleanup=proctoring_alert_policy.clear_session)


# --------------- REST Endpoints ---------------

@router.post("/offer")
async def offer(request: Request):
    """
    WebRTC signaling: browser sends SDP offer, server returns SDP answer.
    Creates a new proctoring session with YOLO video analysis.
    """
    params = await request.json()

    if "sdp" not in params or "type" not in params:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid WebRTC offer. Body must include 'sdp' and 'type'."},
        )

    # Endpoint này có HAI bên gọi, gửi hai bộ metadata khác nhau:
    #
    #   1. client WPF nối thẳng  -> {"exam_attempt_id"}
    #   2. relay từ vox-streaming -> {"sessionId", "scheduleId", "participantId", "streamId", "streamType"}
    #
    # Ở đây có HAI khoá, cố ý tách rời:
    #
    # - session_id là khoá CỤC BỘ của riêng kết nối này: peer connection, sổ sự kiện, SSE và trạng
    #   thái streak/cooldown của alert policy đều tra theo nó, nên nó phải duy nhất theo TỪNG KẾT
    #   NỐI. Một thí sinh relay cả camera lẫn screen là hai kết nối; lấy id phiên thi làm khoá chung
    #   thì kết nối sau đè lên kết nối trước trong session_map, danh tính luồng này gắn nhầm sang
    #   luồng kia, và cooldown của hai luồng trộn làm một.
    # - exam_session_id là khoá ĐỐI NGOẠI, đi kèm mọi cảnh báo bắn sang vox-streaming. Trước đây
    #   đường relay không gửi khoá nào cả nên cảnh báo chạy dưới chính uuid4 cục bộ: nhánh live rơi
    #   vào một kênh Redis không màn hình nào nghe, nhánh durable thì Java tra không ra phiên thi
    #   rồi bỏ qua. Cảnh báo vẫn được phát đủ, chỉ là không tới được ai.
    session_id = str(params.get("exam_attempt_id") or uuid.uuid4())

    # Đuổi kết nối cũ TRƯỚC khi dựng cái mới -- cùng lý do và cùng thứ tự với
    # realtime_controller.py. Máy thi nối lại sau khi rớt mạng gửi lại ĐÚNG exam_attempt_id, tức đúng
    # session_id này, nên nếu không đuổi thì hai peer cùng sống dưới một khoá: peer cũ vẫn giữ handler
    # trỏ vào khoá đó và sẽ dọn dẹp NHẦM peer mới khi nó chết. Xem is_current_connection.
    #
    # Giữ sổ sự kiện và trạng thái alert policy (evict_previous_connection, không phải
    # cleanup_session): nối lại giữa bài không phải là kết thúc phiên thi.
    reconnected = await proctoring_session.evict_previous_connection(session_id)

    # Kiểm tra hạn mức SAU khi đuổi: peer đã chết của chính thí sinh này vẫn nằm trong `pcs` cho tới
    # lúc đó, nên đếm trước khi đuổi có thể từ chối đúng cái kết nối đang đi sửa một sự cố mạng --
    # hỏng đúng vào lúc hệ thống đang bận nhất.
    if len(proctoring_session.pcs) >= MAX_CONNECTIONS:
        return JSONResponse(
            status_code=503,
            content={
                "error": f"Max connections ({MAX_CONNECTIONS}) reached. Try again later.",
                "active_connections": len(proctoring_session.pcs),
            },
        )

    logger.info(
        "[WEBRTC] %s session %s",
        "Nối lại" if reconnected else "Mở",
        session_id,
    )

    pc = RTCPeerConnection(configuration=build_ice_servers())
    proctoring_session.pcs.add(pc)
    proctoring_session.session_map[session_id] = pc
    # Ghi lại đúng những gì bên gọi biết, không hơn. Đường WPF không cầm candidate id nên chỗ đó sẽ
    # rỗng, và vox-streaming sẽ tự tra bù từ session registry của nó.
    proctoring_session.register_identity(
        session_id,
        exam_session_id=params.get("sessionId") or params.get("exam_attempt_id") or "",
        participant_id=params.get("participantId") or "",
        schedule_id=params.get("scheduleId") or "",
        stream_id=params.get("streamId") or "",
        stream_type=params.get("streamType") or "",
    )

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("[WEBRTC] Session %s state: %s", session_id, pc.connectionState)
        if pc.connectionState not in ("failed", "closed", "disconnected"):
            return

        # Chốt danh tính: chỉ peer ĐANG được đăng ký mới được phép dọn dẹp khoá này. Peer đã bị
        # thay thế chết sau đó là chuyện bình thường -- nó vừa bị đuổi -- và nó KHÔNG được kéo theo
        # peer đang thay nó phục vụ. Xem is_current_connection.
        if not proctoring_session.is_current_connection(session_id, pc):
            logger.info(
                "[WEBRTC] Bỏ qua teardown của kết nối đã bị thay thế session=%s state=%s",
                session_id,
                pc.connectionState,
            )
            return

        await _cleanup_session(session_id)

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
        "active_connections": len(proctoring_session.pcs),
        "max_connections": MAX_CONNECTIONS,
        "sessions": list(proctoring_session.session_map.keys()),
    }


@router.get("/connections/{session_id}/events")
def get_session_events(session_id: str, limit: int = 100):
    """Get stored proctoring events for a session."""
    if session_id not in proctoring_session.session_map and session_id not in proctoring_session.session_events:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    events = proctoring_session.session_events.get(session_id, [])
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
    if session_id not in proctoring_session.session_map and session_id not in proctoring_session.session_events:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    queue: asyncio.Queue = asyncio.Queue()
    proctoring_session.session_subscribers[session_id].add(queue)

    async def event_generator():
        try:
            # Send existing events first
            for event in proctoring_session.session_events.get(session_id, []):
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
                    yield f": keepalive {proctoring_session.utc_now()}\n\n"
        finally:
            proctoring_session.session_subscribers[session_id].discard(queue)

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
    if session_id not in proctoring_session.session_map:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    await _cleanup_session(session_id)
    return {"status": "disconnected", "session_id": session_id}
