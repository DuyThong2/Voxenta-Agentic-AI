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
from realtime.attempt.connection import AttemptConnection
from realtime.attempt.registry import (
    register_attempt_connection,
    unregister_attempt_connection,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/realtime", tags=["Realtime"])


def _elapsed_speech_seconds(resume_state: dict | None) -> float:
    state = resume_state or {}
    durations_by_turn: dict[int, float] = {}
    entries = [
        *(state.get("turns") or []),
        *(state.get("realtime_transcripts") or []),
    ]
    for entry in entries:
        try:
            turn_order = int(entry.get("turn_order"))
            duration = max(0.0, float(entry.get("duration_seconds") or 0))
        except (AttributeError, TypeError, ValueError):
            continue
        durations_by_turn[turn_order] = max(
            durations_by_turn.get(turn_order, 0.0),
            duration,
        )

    try:
        checkpointed = max(
            0.0,
            float(state.get("speech_budget_elapsed_seconds") or 0),
        )
    except (TypeError, ValueError):
        checkpointed = 0.0
    return max(checkpointed, sum(durations_by_turn.values()))


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


@router.get("/attempts/{exam_attempt_id}/resume-state")
async def get_attempt_resume_state(
    request: Request,
    exam_attempt_id: str,
    answer_id: str,
):
    """Return the next durable turn for a newly-started client process.

    exam_attempt_id remains part of the route so this endpoint is scoped like
    current-answer. The archive itself is keyed by answer_id, which is the
    durable question identity returned by current-answer.
    """
    resume_state = await archive_store.get_resume_state(
        request.app.state.archive_graph,
        answer_id,
    )
    turns = (resume_state or {}).get("turns") or []
    elapsed_speech_seconds = _elapsed_speech_seconds(resume_state)
    if not turns:
        return {
            "answerId": answer_id,
            "paperItemId": (resume_state or {}).get("paper_item_id"),
            "turnOrder": 1,
            "activePromptText": None,
            "hasFollowUp": False,
            "elapsedSpeechSeconds": elapsed_speech_seconds,
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
        "elapsedSpeechSeconds": elapsed_speech_seconds,
    }


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
    register_attempt_connection(connection)

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
        unregister_attempt_connection(connection)
        await connection.close()
