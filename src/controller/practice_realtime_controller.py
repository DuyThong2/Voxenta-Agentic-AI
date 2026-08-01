"""Realtime practice-session WebSocket endpoint -- parallel to controller/realtime_controller.py
(gói 11 mục 2.5). Reuses the exact same app.state.archive_graph/text_followup_graph as exam
(archive_store's functions are generic over the thread_id string, see
realtime/practice_attempt/session.py's module docstring), just with practice's own id values.
"""

import logging

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from infra.database import archive_store
from infra.realtime_socket import RealtimeSocket
from realtime.practice_attempt.connection import PracticeAttemptConnection
from realtime.practice_attempt.registry import (
    register_practice_attempt_connection,
    unregister_practice_attempt_connection,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/realtime", tags=["Practice realtime"])


@router.get("/practice-sessions/{practice_session_id}/current-answer")
async def get_practice_current_answer(request: Request, practice_session_id: str):
    """Mirror of /attempts/{id}/current-answer -- which answer_id (question) this practice
    session was last on, for a client that lost all local state to find out where to resume
    before opening the WebSocket."""
    answer_id = await archive_store.get_current_answer_id(
        request.app.state.archive_graph, practice_session_id
    )
    return {"answer_id": answer_id}


@router.get("/practice-sessions/{practice_session_id}/resume-state")
async def get_practice_resume_state(
    request: Request,
    practice_session_id: str,
    answer_id: str,
):
    """Mirror of /attempts/{id}/resume-state -- same shape, same semantics, just scoped to a
    practice_session_id/answer_id pair instead of exam_attempt_id/answer_id."""
    resume_state = await archive_store.get_resume_state(
        request.app.state.archive_graph,
        answer_id,
    )
    turns = (resume_state or {}).get("turns") or []
    if not turns:
        return {
            "answerId": answer_id,
            "paperItemId": (resume_state or {}).get("paper_item_id"),
            "turnOrder": 1,
            "activePromptText": None,
            "hasFollowUp": False,
        }

    turn_order = max(int(turn.get("turn_order") or 0) for turn in turns) + 1
    active_prompt_text = resume_state.get("active_prompt_text")
    if not isinstance(active_prompt_text, str) or not active_prompt_text.strip():
        last_prompt_text = turns[-1].get("prompt_text")
        active_prompt_text = (
            last_prompt_text.strip()
            if isinstance(last_prompt_text, str) and last_prompt_text.strip()
            else None
        )
    return {
        "answerId": answer_id,
        "paperItemId": resume_state.get("paper_item_id"),
        "turnOrder": turn_order,
        "activePromptText": active_prompt_text,
        "hasFollowUp": True,
    }


@router.websocket("/practice-sessions/{practice_session_id}")
async def realtime_practice_session_socket(websocket: WebSocket, practice_session_id: str):
    socket = RealtimeSocket(websocket)
    await socket.accept()

    connection = PracticeAttemptConnection(
        practice_session_id=practice_session_id,
        socket=socket,
        archive_graph=websocket.app.state.archive_graph,
        text_followup_graph=websocket.app.state.text_followup_graph,
    )

    try:
        await connection.start()
    except Exception:
        logger.exception(
            "[practice_realtime] failed to start Voice Live session practice_session_id=%s",
            practice_session_id,
        )
        await socket.send_json({"type": "error", "text": "voice_live_start_failed"})
        await socket.close(code=1011)
        return

    logger.info("[practice_realtime] connection opened practice_session_id=%s", practice_session_id)
    register_practice_attempt_connection(connection)

    try:
        async for kind, payload in socket.iter_frames():
            try:
                if kind == "audio":
                    await connection.handle_audio_frame(payload)
                else:
                    await connection.handle_message(payload)
            except Exception:
                logger.exception(
                    "[practice_realtime] error handling message practice_session_id=%s",
                    practice_session_id,
                )
    except WebSocketDisconnect:
        logger.info("[practice_realtime] connection closed practice_session_id=%s", practice_session_id)
    finally:
        unregister_practice_attempt_connection(connection)
        await connection.close()
