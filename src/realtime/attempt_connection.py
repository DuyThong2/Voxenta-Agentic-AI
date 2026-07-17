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
this class schedules _speak_and_notify as a fire-and-forget background task --
never awaited inline, so it never delays the question_start_ack/decision
response already being sent back to the client.

Prototype (task/performance.txt): _speak_and_notify sends a `speak` WS message
(text + rate) instead of calling realtime.avatar_speech.speak -- WPF now
synthesizes via Azure TTS and plays it locally (Services/LocalAvatarSpeaker.cs)
instead of this process synthesizing server-side and streaming it back over
the avatar WebRTC audio track. realtime/avatar_speech.py and the avatar WebRTC
media path (realtime/avatar_webrtc.py) are unused by this path but left in
place -- the avatar WebRTC connection itself still opens per attempt, it just
carries nothing but idle silence now.
"""

import asyncio
import logging
from typing import Any, Optional

from node.followUpDecisionGraph.constants import CLOSING_REPLY, EXAM_FAREWELL_TEXT
from node.state_models import QuestionContext
from infra.database import archive_store
from infra.realtime_socket import RealtimeSocket
from realtime.session import RealtimeExamSession
from infra.voice_live_client import VoiceLiveClient, VoiceLiveServerEvent

logger = logging.getLogger(__name__)


class AttemptConnection:
    """Owns one WebSocket (and one VoiceLiveClient) for the duration of one
    exam attempt. Tracks the currently-active RealtimeExamSession (or None
    between questions) and routes incoming control messages/audio/VAD events
    to it."""

    def __init__(self, *, exam_attempt_id: str, socket: RealtimeSocket, archive_graph: Any, text_followup_graph: Any) -> None:
        self.exam_attempt_id = exam_attempt_id
        self.socket = socket
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
            await self.socket.send_json({"type": "exam_end_ack"})
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
        await archive_store.persist_question_snapshot(
            self.archive_graph, answer_id,
            question=question, paper_item_id=paper_item_id,
            language=language, prompt_text=session_prompt_text,
        )
        # Records this as the exam attempt's current question, independent of answer_id's own
        # checkpoint -- lets a client that lost ALL local state (full app close, not just a WS
        # reconnect) ask "which question was I on" without depending on Kafka's
        # answer-turns-recorded topic (which stays silent until a turn actually completes -- see
        # task/realtime-exam-flow-review.md). Unconditional/idempotent, same as
        # persist_question_snapshot above.
        await archive_store.set_current_answer_id(self.archive_graph, self.exam_attempt_id, answer_id)
        # Unconditional: restores turn history + the pending follow-up
        # prompt from the durable archive if this answer_id already has some
        # (reconnect / pod restart mid-question on EKS); a no-op for a
        # genuinely new question. See RealtimeExamSession.hydrate_from_archive.
        await self.active_session.hydrate_from_archive()

        logger.info(
            "[attempt_connection] question_start exam_attempt_id=%s answer_id=%s",
            self.exam_attempt_id, answer_id,
        )
        await self.socket.send_json({"type": "question_start_ack", "answer_id": answer_id})
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
        is_last_allowed_turn = bool(message.get("is_last_allowed_turn"))
        raw_duration_seconds = message.get("duration_seconds")
        try:
            duration_seconds = float(raw_duration_seconds) if raw_duration_seconds is not None else None
        except (TypeError, ValueError):
            duration_seconds = None

        # Fire-and-forget: never awaited inline with the turn_end decision response below.
        # Lets eval-time scoring prefer this (Voice-Live handles code-switched Vietnamese
        # better than the Speech SDK's re-transcription) -- see start_node_config.py.
        # is_last_allowed_turn is persisted here too so _recover_pending_decision can re-apply
        # WPF's MaxTurnsPerQuestion clamp correctly if this turn's decision has to be recomputed
        # during a later resume (see archive_store.persist_realtime_transcript's docstring).
        asyncio.create_task(
            archive_store.persist_realtime_transcript(
                self.archive_graph, session.answer_id, session.turn_order, transcript,
                is_last_allowed_turn=is_last_allowed_turn,
                duration_seconds=duration_seconds,
            )
        )

        word_count = message.get("word_count")
        if word_count is None:
            word_count = len(transcript.split()) if transcript else 0

        # decide_next_step ultimately calls ChatOpenAI.invoke() (sync) -- running it inline here
        # would block this process's single asyncio event loop for the full LLM round-trip,
        # freezing every other exam_attempt's WebRTC (avatar/proctoring) and WS traffic along with
        # it. Confirmed live: under degraded network this call took >20s and the avatar WebRTC
        # peer connection was declared failed mid-call because no RTP/keepalive could be sent
        # while the loop was blocked. asyncio.to_thread keeps this a plain sync call while freeing
        # the loop for everything else, exactly like archive_controller.py already does for
        # archive_graph.invoke.
        decision = await asyncio.to_thread(session.decide_next_step, transcript, word_count, duration_seconds)
        completed_turn_order = session.turn_order - 1
        if is_last_allowed_turn and decision.get("should_continue"):
            # WPF's own MaxTurnsPerQuestion is about to force this question closed regardless of
            # what we decide -- if we still spoke a follow-up here, the student would hear a
            # question read aloud and then get bounced to the next one before ever answering it
            # (confirmed live: this is exactly what was happening). Match WPF's outcome instead of
            # racing it.
            decision = {**decision, "should_continue": False, "next_prompt_text": None, "reason": "client_max_turns_reached"}

        # MUST run on this event-loop thread (not inside the asyncio.to_thread call above) --
        # schedule_publish's asyncio.create_task raises "no running event loop" otherwise. Uses
        # the already-clamped decision so what's persisted (and what a later resume recovers)
        # matches what was actually decided, not the pre-clamp raw graph output.
        session.schedule_publish(
            completed_turn_order,
            reason=decision.get("reason", ""),
            should_continue=bool(decision.get("should_continue")),
            next_prompt_text=decision.get("next_prompt_text"),
        )

        await self.socket.send_json({
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
        """Prototype (task/performance.txt): sends the text to speak over the realtime WS instead
        of synthesizing it here and streaming it back over the avatar WebRTC audio track
        (realtime/avatar_speech.py, now unused by this path). WPF's RealtimeSessionClient
        synthesizes+plays it locally (Services/LocalAvatarSpeaker.cs) and raises its own
        OnAvatarUtteranceComplete once local playback finishes -- this process no longer needs to
        wait for or confirm playback, so this just fires the message and returns."""
        session = self.active_session
        logger.info(
            "[realtime_ai_speech] exam_attempt_id=%s answer_id=%s turn_order=%s sequence=%d text=%r",
            self.exam_attempt_id,
            session.answer_id if session else None,
            session.turn_order if session else None,
            sequence, text,
        )
        await self.socket.send_json({
            "type": "speak",
            "sequence": sequence,
            "text": text or "",
            "rate": "-20%" if slow else None,
        })

    async def _handle_resume(self, message: dict) -> None:
        """Rebuilds self.active_session purely from the durable archive
        (see RealtimeExamSession.create_from_archive) -- the client sends
        only answer_id/turn_order here, no question data, since the question
        snapshot was already persisted durably at question_start
        (_persist_question_snapshot). Restores turn history, turn_order, and
        the pending follow-up prompt, then re-speaks that prompt so the exam
        actually continues instead of leaving the student in silence with a
        rebuilt-but-mute session.

        Also recovers whatever decision a client's still-pending
        SendTurnEndAndWaitAsync might be waiting on (see
        _recover_pending_decision) and hands it back in resume_ack -- this is
        what unblocks a client left awaiting a decision that never arrived
        because the connection dropped between turn_end being sent and the
        response reaching it (task/exam-interrupted-session-grading.txt)."""
        answer_id = message.get("answer_id")
        session = await RealtimeExamSession.create_from_archive(
            answer_id=answer_id,
            exam_attempt_id=self.exam_attempt_id,
            archive_graph=self.archive_graph,
            graph=self.text_followup_graph,
        )

        last_archived_turn_order = 0
        recovered_decision: Optional[dict] = None
        recovered_turn_order: Optional[int] = None
        if session is not None:
            self.active_session = session
            last_archived_turn_order = session.turn_order - 1
            recovered_decision, recovered_turn_order = await self._recover_pending_decision(session)
            logger.info(
                "[attempt_connection] resume rebuilt session exam_attempt_id=%s answer_id=%s turn_order=%d recovered_turn_order=%s",
                self.exam_attempt_id, answer_id, session.turn_order, recovered_turn_order,
            )
        else:
            logger.warning(
                "[attempt_connection] resume with nothing archived for answer_id=%s exam_attempt_id=%s",
                answer_id, self.exam_attempt_id,
            )

        ack_payload: dict = {
            "type": "resume_ack",
            "answer_id": answer_id,
            "last_archived_turn_order": last_archived_turn_order,
        }
        if recovered_decision is not None:
            ack_payload["recovered_turn_order"] = recovered_turn_order
            ack_payload["decision"] = recovered_decision
        await self.socket.send_json(ack_payload)

        if recovered_decision is not None:
            # A recovered should_continue=False means the question actually finished --
            # session.current_prompt_text was NOT updated for that outcome (decide_next_step only
            # updates it when continuing) and still holds the stale PREVIOUS prompt, so speaking
            # it here would incorrectly re-ask an already-answered follow-up. Mirror
            # _handle_turn_end's own next_prompt_text/CLOSING_REPLY logic instead of falling
            # through to the plain current_prompt_text re-speak below.
            next_prompt_text = recovered_decision.get("next_prompt_text") or (
                None if recovered_decision.get("should_continue") else CLOSING_REPLY
            )
            self._speak(next_prompt_text)
        elif session is not None and session.current_prompt_text:
            self._speak(session.current_prompt_text)

    async def _recover_pending_decision(
        self, session: RealtimeExamSession,
    ) -> tuple[Optional[dict], Optional[int]]:
        """Reconstructs whatever decision a client's still-pending
        SendTurnEndAndWaitAsync might be waiting on, so resume_ack can hand it
        back directly instead of leaving the client to wait for a WS reply
        that already came and went (or never will). Two distinct cases,
        mutually exclusive by construction (they cover different turn_orders)
        -- checked in this order:

        1. A turn_end whose transcript was durably captured
           (archive_store.persist_realtime_transcript, called right before
           the slow decide_next_step call in _handle_turn_end) but whose
           decision was never completed/persisted -- i.e. the connection
           dropped mid-LLM-call, before decide_next_step finished.
           session.turn_order (set by _apply_resume_state from the max
           completed turn_order + 1) is exactly the first turn_order that has
           NOT completed yet, so that's the only one worth checking. Re-runs
           the decision now using the durably-saved transcript (no live
           audio/VAD needed) and persists it exactly like a normal turn_end
           would.
        2. The much more likely case: the turn completed and was persisted
           normally (decide_next_step doesn't depend on the client connection
           at all -- only the final `await self.socket.send_json(...)` in
           _handle_turn_end does), but that reply itself never reached the
           client. Nothing needs recomputing here -- the answer is just the
           last entry in session.turns (already restored from the durable
           archive by _apply_resume_state), at turn_order ==
           last_archived_turn_order.

        Returns (None, None) only if neither case has anything to offer (a
        genuinely fresh resume with nothing outstanding)."""
        pending_turn_order = session.turn_order
        resume_state = await archive_store.get_resume_state(self.archive_graph, session.answer_id)
        pending_entry = None
        if resume_state is not None:
            transcripts = resume_state.get("realtime_transcripts") or []
            pending_entry = next(
                (t for t in transcripts if t.get("turn_order") == pending_turn_order), None,
            )

        if pending_entry is not None:
            transcript = pending_entry.get("text") or ""
            word_count = len(transcript.split()) if transcript else 0
            is_last_allowed_turn = bool(pending_entry.get("is_last_allowed_turn"))
            raw_duration_seconds = pending_entry.get("duration_seconds")
            try:
                duration_seconds = float(raw_duration_seconds) if raw_duration_seconds is not None else None
            except (TypeError, ValueError):
                duration_seconds = None

            logger.info(
                "[attempt_connection] recovering dangling decision exam_attempt_id=%s answer_id=%s turn_order=%d",
                self.exam_attempt_id, session.answer_id, pending_turn_order,
            )
            decision = await asyncio.to_thread(session.decide_next_step, transcript, word_count, duration_seconds)
            if is_last_allowed_turn and decision.get("should_continue"):
                decision = {**decision, "should_continue": False, "next_prompt_text": None, "reason": "client_max_turns_reached"}

            session.schedule_publish(
                pending_turn_order,
                reason=decision.get("reason", ""),
                should_continue=bool(decision.get("should_continue")),
                next_prompt_text=decision.get("next_prompt_text"),
            )
            return decision, pending_turn_order

        if session.turns:
            last_turn = session.turns[-1]
            last_turn_order = last_turn.get("turn_order")
            if last_turn_order is not None:
                logger.info(
                    "[attempt_connection] handing back already-completed decision exam_attempt_id=%s answer_id=%s turn_order=%s",
                    self.exam_attempt_id, session.answer_id, last_turn_order,
                )
                decision = {
                    "should_continue": bool(last_turn.get("should_continue")),
                    "next_prompt_text": last_turn.get("next_prompt_text"),
                    "reason": last_turn.get("decision_reason", ""),
                }
                return decision, int(last_turn_order)

        return None, None

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
        await self.socket.send_json(payload)

    async def close(self) -> None:
        """Best-effort local cleanup when the connection drops. Durable
        state (archived turns, published markers) lives in Postgres via
        archive_graph and is untouched by this — only in-memory session
        state for the question in flight is discarded, exactly as the plan's
        "Path A holds no durable state of its own" design intends."""
        self.active_session = None
        await self.voice_live_client.close()
