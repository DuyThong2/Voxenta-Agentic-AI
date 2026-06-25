"""AttemptConnection: one instance per exam_attempt_id, owning the single
WebSocket that lives for the whole exam attempt (Phase 2 of
docs/realtime-self-hosted-avatar-plan.md).

Unlike a naive per-question design (and unlike the old Tavus integration,
which restarted its conversation per question), the WebSocket here is opened
once per exam attempt and never reconnects between questions — switching
questions is just an in-band `question_start` control message. This class is
the layer that owns that connection; RealtimeExamSession (one per question)
is created/destroyed underneath it as questions advance.

This phase only implements the text-based placeholder protocol described in
the plan doc's Phase 2 section (`question_start`, `transcript_chunk`,
`turn_end`, `resume`). Real binary audio streaming + Azure Voice Live
integration is Phase 3 — not built here.
"""

import logging
from typing import Any, Optional

from fastapi import WebSocket

from node.state_models import QuestionContext
from realtime import turn_publisher
from realtime.session import RealtimeExamSession

logger = logging.getLogger(__name__)


class AttemptConnection:
    """Owns one WebSocket for the duration of one exam attempt. Tracks the
    currently-active RealtimeExamSession (or None between questions) and
    routes incoming control messages to it."""

    def __init__(self, *, exam_attempt_id: str, websocket: WebSocket, archive_graph: Any, text_followup_graph: Any) -> None:
        self.exam_attempt_id = exam_attempt_id
        self.websocket = websocket
        self.archive_graph = archive_graph
        self.text_followup_graph = text_followup_graph
        self.active_session: Optional[RealtimeExamSession] = None

    async def handle_message(self, message: dict) -> None:
        message_type = message.get("type")
        if message_type == "question_start":
            await self._handle_question_start(message)
        elif message_type == "transcript_chunk":
            await self._handle_transcript_chunk(message)
        elif message_type == "turn_end":
            await self._handle_turn_end(message)
        elif message_type == "resume":
            await self._handle_resume(message)
        else:
            logger.warning(
                "[attempt_connection] unknown message type=%s exam_attempt_id=%s",
                message_type, self.exam_attempt_id,
            )

    async def _handle_question_start(self, message: dict) -> None:
        answer_id = message.get("answer_id")
        question_payload = message.get("question") or {}
        question = QuestionContext.model_validate(question_payload) if question_payload else None
        prompt_text = question_payload.get("question_text") if isinstance(question_payload, dict) else None
        language = message.get("language", "en-US")

        # Replaces any previous session outright — a new question always
        # starts a fresh RealtimeExamSession with turn_order reset to 1,
        # even if the previous question's session never saw a turn_end.
        # text_followup_graph has no checkpointer (see build_text_followup_graph),
        # so sharing the one instance across sessions/questions is safe.
        self.active_session = RealtimeExamSession(
            answer_id=answer_id,
            question=question,
            prompt_text=prompt_text,
            language=language,
            archive_graph=self.archive_graph,
            graph=self.text_followup_graph,
        )

        logger.info(
            "[attempt_connection] question_start exam_attempt_id=%s answer_id=%s",
            self.exam_attempt_id, answer_id,
        )
        await self.websocket.send_json({"type": "question_start_ack", "answer_id": answer_id})

    async def _handle_transcript_chunk(self, message: dict) -> None:
        if self.active_session is None:
            logger.warning(
                "[attempt_connection] transcript_chunk with no active session exam_attempt_id=%s",
                self.exam_attempt_id,
            )
            return
        self.active_session.append_transcript_chunk(message.get("text", ""))

    async def _handle_turn_end(self, message: dict) -> None:
        if self.active_session is None:
            logger.warning(
                "[attempt_connection] turn_end with no active session exam_attempt_id=%s",
                self.exam_attempt_id,
            )
            return

        session = self.active_session
        transcript = message.get("text", session.current_transcript)
        word_count = message.get("word_count")
        if word_count is None:
            word_count = len(transcript.split()) if transcript else 0

        decision = session.decide_next_step(transcript, word_count)
        await self.websocket.send_json({
            "type": "decision",
            "answer_id": session.answer_id,
            "decision": decision,
        })

    async def _handle_resume(self, message: dict) -> None:
        answer_id = message.get("answer_id")
        last_archived_turn_order = await turn_publisher.get_last_archived_turn_order(
            self.archive_graph, answer_id,
        )
        await self.websocket.send_json({
            "type": "resume_ack",
            "answer_id": answer_id,
            "last_archived_turn_order": last_archived_turn_order,
        })

    def close(self) -> None:
        """Best-effort local cleanup when the connection drops. Durable
        state (archived turns, published markers) lives in Postgres via
        archive_graph and is untouched by this — only in-memory session
        state for the question in flight is discarded, exactly as the plan's
        "Path A holds no durable state of its own" design intends."""
        self.active_session = None
