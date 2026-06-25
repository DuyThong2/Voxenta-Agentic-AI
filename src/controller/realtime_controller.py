"""Realtime exam WebSocket endpoint (Phase 2 of
docs/realtime-self-hosted-avatar-plan.md).

One WebSocket connection per exam attempt, not per question — this is the
direct fix for Tavus's "fresh conversation per question" reconnect-gap
problem. AttemptConnection owns the connection for its lifetime and
creates/destroys a RealtimeExamSession per question as question_start
control messages arrive.

This phase only implements the text-based placeholder protocol; real binary
audio streaming (Azure Voice Live) is Phase 3.
"""

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

    logger.info("[realtime] connection opened exam_attempt_id=%s", exam_attempt_id)

    try:
        while True:
            message = await websocket.receive_json()
            try:
                await connection.handle_message(message)
            except Exception:
                logger.exception(
                    "[realtime] error handling message exam_attempt_id=%s message_type=%s",
                    exam_attempt_id, message.get("type"),
                )
    except WebSocketDisconnect:
        logger.info("[realtime] connection closed exam_attempt_id=%s", exam_attempt_id)
    finally:
        connection.close()
