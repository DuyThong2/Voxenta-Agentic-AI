"""Realtime exam WebSocket endpoint (Phase 2-3 of
docs/realtime-self-hosted-avatar-plan.md).

One WebSocket connection per exam attempt, not per question — this is the
direct fix for Tavus's "fresh conversation per question" reconnect-gap
problem. AttemptConnection owns the connection (and, since Phase 3, the one
Azure Voice Live session) for its lifetime and creates/destroys a
RealtimeExamSession per question as question_start control messages arrive.

Frames are mixed: binary frames are raw PCM16 mic audio (routed to Voice
Live via AttemptConnection.handle_audio_frame); text frames are the JSON
control protocol (question_start/turn_end/resume) plus the VAD/transcript
events AttemptConnection forwards back out.
"""

import logging

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from infra.database import archive_store
from infra.realtime_socket import RealtimeSocket
from realtime.attempt_connection import AttemptConnection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/realtime", tags=["Realtime"])


@router.get("/attempts/{exam_attempt_id}/current-answer")
async def get_current_answer(request: Request, exam_attempt_id: str):
    """Which answer_id (question) exam_attempt_id was last on, per
    archive_store.set_current_answer_id -- for a client that lost all local state (full app
    close, not just a WS reconnect) to find out where to resume before it even opens the realtime
    WebSocket. answer_id is null if this attempt never started any question yet (see
    task/realtime-exam-flow-review.md for why this doesn't rely on Kafka's
    answer-turns-recorded topic)."""
    answer_id = await archive_store.get_current_answer_id(request.app.state.archive_graph, exam_attempt_id)
    return {"answer_id": answer_id}


@router.websocket("/attempts/{exam_attempt_id}")
async def realtime_attempt_socket(websocket: WebSocket, exam_attempt_id: str):
    socket = RealtimeSocket(websocket)
    await socket.accept()

    connection = AttemptConnection(
        exam_attempt_id=exam_attempt_id,
        socket=socket,
        archive_graph=websocket.app.state.archive_graph,
        text_followup_graph=websocket.app.state.text_followup_graph,
    )

    try:
        await connection.start()
    except Exception:
        logger.exception("[realtime] failed to start Voice Live session exam_attempt_id=%s", exam_attempt_id)
        await socket.send_json({"type": "error", "text": "voice_live_start_failed"})
        await socket.close(code=1011)
        return

    logger.info("[realtime] connection opened exam_attempt_id=%s", exam_attempt_id)

    try:
        async for kind, payload in socket.iter_frames():
            try:
                if kind == "audio":
                    await connection.handle_audio_frame(payload)
                else:
                    await connection.handle_message(payload)
            except Exception:
                logger.exception(
                    "[realtime] error handling message exam_attempt_id=%s",
                    exam_attempt_id,
                )
    except WebSocketDisconnect:
        logger.info("[realtime] connection closed exam_attempt_id=%s", exam_attempt_id)
    finally:
        await connection.close()
