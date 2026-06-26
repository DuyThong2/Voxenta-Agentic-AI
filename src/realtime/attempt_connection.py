"""AttemptConnection: one instance per exam_attempt_id, owning the single
WebSocket that lives for the whole exam attempt (Phase 2 of
docs/realtime-self-hosted-avatar-plan.md).

Unlike a naive per-question design (and unlike the old Tavus integration,
which restarted its conversation per question), the WebSocket here is opened
once per exam attempt and never reconnects between questions — switching
questions is just an in-band `question_start` control message. This class is
the layer that owns that connection; RealtimeExamSession (one per question)
is created/destroyed underneath it as questions advance.

Phase 3: also owns the one VoiceLiveClient for the whole attempt (mirrors the
WebSocket's own per-attempt lifetime — Voice Live's per-utterance
conversation items naturally segment multiple questions' speech without a
reconnect). Binary WebSocket frames are raw PCM16 mic audio, forwarded
straight into VoiceLiveClient; text frames are the JSON control protocol
(`question_start`, `turn_end`, `resume`) plus VAD/transcript events forwarded
back out to the client.

Phase 4: whenever the avatar needs to speak (the question prompt at
question_start, or decision.next_prompt_text/CLOSING_REPLY after a turn_end),
this class schedules realtime.avatar_speech.speak as a fire-and-forget
background task -- never awaited inline, so a slow render never delays the
question_start_ack/decision response already being sent back to the client.
"""

import asyncio
import logging
from typing import Any, Optional

from fastapi import WebSocket

from node.followUpDecisionGraph.constants import CLOSING_REPLY
from node.state_models import QuestionContext
from realtime import avatar_speech, turn_publisher
from realtime.session import RealtimeExamSession
from realtime.voice_live_client import VoiceLiveClient, VoiceLiveServerEvent

logger = logging.getLogger(__name__)


class AttemptConnection:
    """Owns one WebSocket (and one VoiceLiveClient) for the duration of one
    exam attempt. Tracks the currently-active RealtimeExamSession (or None
    between questions) and routes incoming control messages/audio/VAD events
    to it."""

    def __init__(self, *, exam_attempt_id: str, websocket: WebSocket, archive_graph: Any, text_followup_graph: Any) -> None:
        self.exam_attempt_id = exam_attempt_id
        self.websocket = websocket
        self.archive_graph = archive_graph
        self.text_followup_graph = text_followup_graph
        self.active_session: Optional[RealtimeExamSession] = None
        self.voice_live_client = VoiceLiveClient(on_event=self._on_voice_live_event)
        self._utterance_sequence = 0

    async def start(self) -> None:
        await self.voice_live_client.start()

    async def handle_audio_frame(self, data: bytes) -> None:
        await self.voice_live_client.push_audio(data)

    async def handle_message(self, message: dict) -> None:
        message_type = message.get("type")
        if message_type == "question_start":
            await self._handle_question_start(message)
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
        self._speak(prompt_text)

    async def _handle_turn_end(self, message: dict) -> None:
        if self.active_session is None:
            logger.warning(
                "[attempt_connection] turn_end with no active session exam_attempt_id=%s",
                self.exam_attempt_id,
            )
            return

        session = self.active_session
        explicit_text = message.get("text")
        if explicit_text is not None:
            # Manual override (test clients / the old text protocol) — skip the VAD wait below
            # since the caller is providing the transcript directly.
            transcript = explicit_text
        else:
            # turn_end can arrive right on the heels of vad_speech_end, before that utterance's
            # (slightly slower) final_transcript has landed — wait briefly so this turn's
            # decision isn't made on an incomplete transcript.
            await session.wait_for_pending_transcript()
            transcript = session.current_transcript

        word_count = message.get("word_count")
        if word_count is None:
            word_count = len(transcript.split()) if transcript else 0

        decision = session.decide_next_step(transcript, word_count)
        await self.websocket.send_json({
            "type": "decision",
            "answer_id": session.answer_id,
            "decision": decision,
        })
        next_prompt_text = decision.get("next_prompt_text") or (
            None if decision.get("should_continue") else CLOSING_REPLY
        )
        self._speak(next_prompt_text)

    def _speak(self, text: Optional[str]) -> None:
        if not text:
            return
        self._utterance_sequence += 1
        asyncio.create_task(
            avatar_speech.speak(self.exam_attempt_id, text, sequence=self._utterance_sequence)
        )

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

    async def _on_voice_live_event(self, event: VoiceLiveServerEvent) -> None:
        """Routes a translated Voice Live event to the active session's
        transcript-accumulation state and forwards it to the WPF client (for
        UI feedback and so the client can decide to send turn_end on
        vad_speech_end)."""
        if event.kind == "vad_speech_start" and self.active_session is not None:
            self.active_session.on_speech_start()
        elif event.kind == "partial_transcript" and self.active_session is not None:
            self.active_session.on_partial_transcript(event.text)
        elif event.kind == "vad_speech_end" and self.active_session is not None:
            self.active_session.on_speech_end()
        elif event.kind == "final_transcript" and self.active_session is not None:
            self.active_session.on_final_transcript(event.text)
        elif event.kind in ("vad_speech_start", "vad_speech_end", "partial_transcript", "final_transcript"):
            logger.warning(
                "[attempt_connection] voice_live event=%s with no active session exam_attempt_id=%s",
                event.kind, self.exam_attempt_id,
            )

        payload = {"type": event.kind}
        if event.text is not None:
            payload["text"] = event.text
        try:
            await self.websocket.send_json(payload)
        except Exception:
            logger.exception(
                "[attempt_connection] failed to forward voice_live event=%s exam_attempt_id=%s",
                event.kind, self.exam_attempt_id,
            )

    async def close(self) -> None:
        """Best-effort local cleanup when the connection drops. Durable
        state (archived turns, published markers) lives in Postgres via
        archive_graph and is untouched by this — only in-memory session
        state for the question in flight is discarded, exactly as the plan's
        "Path A holds no durable state of its own" design intends."""
        self.active_session = None
        await self.voice_live_client.close()
