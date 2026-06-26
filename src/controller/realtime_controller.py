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

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from realtime.attempt_connection import AttemptConnection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/realtime", tags=["Realtime"])


@router.websocket("/attempts/{exam_attempt_id}")
async def realtime_attempt_socket(websocket: WebSocket, exam_attempt_id: str):
    await websocket.accept()

    connection = AttemptConnection(
        exam_attempt_id=exam_attempt_id,
        websocket=websocket,
        archive_graph=websocket.app.state.archive_graph,
        text_followup_graph=websocket.app.state.text_followup_graph,
    )

    try:
        await connection.start()
    except Exception:
        logger.exception("[realtime] failed to start Voice Live session exam_attempt_id=%s", exam_attempt_id)
        await websocket.send_json({"type": "error", "text": "voice_live_start_failed"})
        await websocket.close(code=1011)
        return

    logger.info("[realtime] connection opened exam_attempt_id=%s", exam_attempt_id)

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000), message.get("reason"))

            try:
                if message.get("bytes") is not None:
                    await connection.handle_audio_frame(message["bytes"])
                    continue

                text = message.get("text")
                if text is None:
                    continue
                parsed = json.loads(text)
                await connection.handle_message(parsed)
            except Exception:
                logger.exception(
                    "[realtime] error handling message exam_attempt_id=%s",
                    exam_attempt_id,
                )
    except WebSocketDisconnect:
        logger.info("[realtime] connection closed exam_attempt_id=%s", exam_attempt_id)
    finally:
        await connection.close()
