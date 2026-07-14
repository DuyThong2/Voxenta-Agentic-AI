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

from node.followUpDecisionGraph.constants import CLOSING_REPLY, EXAM_FAREWELL_TEXT
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
        elif message_type == "present_question":
            await self._handle_present_question(message)
        elif message_type == "turn_end":
            await self._handle_turn_end(message)
        elif message_type == "resume":
            await self._handle_resume(message)
        elif message_type == "exam_end":
            self._speak(EXAM_FAREWELL_TEXT)
            await self.websocket.send_json({"type": "exam_end_ack"})
        else:
            logger.warning(
                "[attempt_connection] unknown message type=%s exam_attempt_id=%s",
                message_type, self.exam_attempt_id,
            )

    async def _handle_question_start(self, message: dict) -> None:
        answer_id = message.get("answer_id")
        paper_item_id = message.get("paper_item_id")
        question_payload = message.get("question") or {}
        question = QuestionContext.model_validate(question_payload) if question_payload else None
        session_prompt_text = question_payload.get("question_text") if isinstance(question_payload, dict) else None
        section_instruction = message.get("section_instruction")
        language = message.get("language", "en-US")

        # Replaces any previous session outright — a new question always
        # starts a fresh RealtimeExamSession with turn_order reset to 1,
        # even if the previous question's session never saw a turn_end.
        # text_followup_graph has no checkpointer (see build_text_followup_graph),
        # so sharing the one instance across sessions/questions is safe.
        self.active_session = RealtimeExamSession(
            answer_id=answer_id,
            exam_attempt_id=self.exam_attempt_id,
            paper_item_id=paper_item_id,
            question=question,
            prompt_text=session_prompt_text,
            language=language,
            archive_graph=self.archive_graph,
            graph=self.text_followup_graph,
        )
        # Durably snapshot the question payload for this answer_id, once --
        # this is what lets a later `resume` (see _handle_resume) rebuild a
        # full session from nothing but answer_id, with no question data
        # resent by the client. Safe/idempotent: a genuinely new question
        # writes fresh values; the same question re-sent via question_start
        # (rather than resume) overwrites with identical values.
        await turn_publisher.persist_question_snapshot(
            self.archive_graph, answer_id,
            question=question, paper_item_id=paper_item_id,
            language=language, prompt_text=session_prompt_text,
        )
        # Unconditional: restores turn history + the pending follow-up
        # prompt from the durable archive if this answer_id already has some
        # (reconnect / pod restart mid-question on EKS); a no-op for a
        # genuinely new question. See RealtimeExamSession.hydrate_from_archive.
        await self.active_session.hydrate_from_archive()

        logger.info(
            "[attempt_connection] question_start exam_attempt_id=%s answer_id=%s",
            self.exam_attempt_id, answer_id,
        )
        await self.websocket.send_json({"type": "question_start_ack", "answer_id": answer_id})
        # question_start only speaks the section-level lead-in (when this question starts a new
        # section). instruction_text and the question prompt are spoken via separate
        # present_question messages the WPF client sends afterwards -- with a deliberate pause and
        # concurrent asset display in between -- so this stays a single, short utterance instead of
        # bundling section + instruction + prompt together. See
        # RealtimeExamFlowService.cs::RunQuestionAsync for the sequencing this pairs with.
        self._speak((section_instruction or "").strip())

    async def _handle_present_question(self, message: dict) -> None:
        prompt_text = message.get("prompt_text")
        if self.active_session is not None and prompt_text:
            self.active_session.current_prompt_text = prompt_text
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

        logger.info(
            "[realtime_transcript] exam_attempt_id=%s answer_id=%s turn_order=%d text=%r",
            self.exam_attempt_id, session.answer_id, session.turn_order, transcript,
        )
        # Fire-and-forget: never awaited inline with the turn_end decision response below.
        # Lets eval-time scoring prefer this (Voice-Live handles code-switched Vietnamese
        # better than the Speech SDK's re-transcription) -- see start_node_config.py.
        asyncio.create_task(
            turn_publisher.persist_realtime_transcript(
                self.archive_graph, session.answer_id, session.turn_order, transcript,
            )
        )

        word_count = message.get("word_count")
        if word_count is None:
            word_count = len(transcript.split()) if transcript else 0

        decision = session.decide_next_step(transcript, word_count)
        if message.get("is_last_allowed_turn") and decision.get("should_continue"):
            # WPF's own MaxTurnsPerQuestion is about to force this question closed regardless of
            # what we decide -- if we still spoke a follow-up here, the student would hear a
            # question read aloud and then get bounced to the next one before ever answering it
            # (confirmed live: this is exactly what was happening). Match WPF's outcome instead of
            # racing it.
            decision = {**decision, "should_continue": False, "next_prompt_text": None, "reason": "client_max_turns_reached"}
        await self.websocket.send_json({
            "type": "decision",
            "answer_id": session.answer_id,
            "decision": decision,
        })
        next_prompt_text = decision.get("next_prompt_text") or (
            None if decision.get("should_continue") else CLOSING_REPLY
        )
        self._speak(
            next_prompt_text,
            slow=decision.get("reason") in {
                "clarify_prompt",
                "decline_repair",
            },
        )

    def _speak(self, text: Optional[str], *, slow: bool = False) -> None:
        self._utterance_sequence += 1
        asyncio.create_task(self._speak_and_notify(text, self._utterance_sequence, slow))

    async def _speak_and_notify(self, text: Optional[str], sequence: int, slow: bool) -> None:
        try:
            if text:
                session = self.active_session
                logger.info(
                    "[realtime_ai_speech] exam_attempt_id=%s answer_id=%s turn_order=%s sequence=%d text=%r",
                    self.exam_attempt_id,
                    session.answer_id if session else None,
                    session.turn_order if session else None,
                    sequence, text,
                )
                await avatar_speech.speak(
                    self.exam_attempt_id,
                    text,
                    sequence=sequence,
                    rate="-20%" if slow else None,
                )
        finally:
            try:
                await self.websocket.send_json({
                    "type": "avatar_utterance_complete",
                    "sequence": sequence,
                    "text": text or "",
                })
            except Exception:
                logger.exception(
                    "[attempt_connection] failed to send avatar_utterance_complete exam_attempt_id=%s sequence=%d",
                    self.exam_attempt_id, sequence,
                )

    async def _handle_resume(self, message: dict) -> None:
        """Rebuilds self.active_session purely from the durable archive
        (see RealtimeExamSession.create_from_archive) -- the client sends
        only answer_id/turn_order here, no question data, since the question
        snapshot was already persisted durably at question_start
        (_persist_question_snapshot). Restores turn history, turn_order, and
        the pending follow-up prompt, then re-speaks that prompt so the exam
        actually continues instead of leaving the student in silence with a
        rebuilt-but-mute session."""
        answer_id = message.get("answer_id")
        session = await RealtimeExamSession.create_from_archive(
            answer_id=answer_id,
            exam_attempt_id=self.exam_attempt_id,
            archive_graph=self.archive_graph,
            graph=self.text_followup_graph,
        )

        last_archived_turn_order = 0
        if session is not None:
            self.active_session = session
            last_archived_turn_order = session.turn_order - 1
            logger.info(
                "[attempt_connection] resume rebuilt session exam_attempt_id=%s answer_id=%s turn_order=%d",
                self.exam_attempt_id, answer_id, session.turn_order,
            )
        else:
            logger.warning(
                "[attempt_connection] resume with nothing archived for answer_id=%s exam_attempt_id=%s",
                answer_id, self.exam_attempt_id,
            )

        await self.websocket.send_json({
            "type": "resume_ack",
            "answer_id": answer_id,
            "last_archived_turn_order": last_archived_turn_order,
        })

        if session is not None and session.current_prompt_text:
            self._speak(session.current_prompt_text)

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
