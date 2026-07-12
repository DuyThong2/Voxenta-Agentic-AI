"""RealtimeExamSession: one instance per answer_id/question, created and
destroyed by AttemptConnection (Phase 2 of
docs/realtime-self-hosted-avatar-plan.md).

Unlike the old Tavus endpoint (controller/tavus_controller.py), which had to
reconstruct FollowUpGraphState by parsing an OpenAI-style message history
(see mappers/chat_completion_mapper.build_followup_state_from_messages),
this session already holds its own turn-accumulation state directly (it's a
long-lived Python object scoped to one question, not a stateless HTTP
handler), so it can build the FollowUpGraphState input straight from
attributes — no message-format parsing needed.

Path A (decision) / Path B (archive + Kafka) decoupling: decide_next_step
returns the decision dict immediately and only *schedules*
turn_publisher.publish_turn_if_new as a fire-and-forget background task; it
never awaits it inline, so a slow/racing archive never delays the decision
the client is waiting on.

Transcript accumulation (Phase 3): live transcript text comes from Azure
Voice Live's VAD-delimited utterances (see voice_live_client.py), not the
Phase 2 placeholder's manually-typed `transcript_chunk` messages. A turn can
contain more than one utterance (the speaker pauses then resumes before
WPF decides the turn is actually over), so each utterance's *final*
transcript is appended to `current_transcript` as it completes, not
replaced — only the in-progress utterance's partial text is held separately
until its own final_transcript arrives.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from node.followUpDecisionGraph.graphConfig import build_text_followup_graph
from node.state_models import QuestionContext
from realtime import turn_publisher

logger = logging.getLogger(__name__)

# How long to wait, after a turn_end is requested, for a VAD-detected utterance's final_transcript
# to arrive if speech_end already fired but the (slower) transcription hasn't completed yet.
# Mirrors the margin spikes/voice_live_poc.py needed (final_transcript landed ~0.85s after
# speech_end in real testing) -- chosen with headroom, not measured precisely against production
# load.
PENDING_TRANSCRIPT_TIMEOUT_SECONDS = 3.0


class RealtimeExamSession:
    """Holds turn-accumulation state for exactly one question
    (answer_id). turn_order starts at 1 and increments per turn within this
    session; a new question always gets a brand-new RealtimeExamSession with
    turn_order reset to 1, created by AttemptConnection."""

    def __init__(
        self,
        *,
        answer_id: str,
        exam_attempt_id: str,
        paper_item_id: Optional[str],
        question: Optional[QuestionContext],
        prompt_text: Optional[str],
        language: str,
        archive_graph: Any,
        graph=None,
    ) -> None:
        self.answer_id = answer_id
        self.exam_attempt_id = exam_attempt_id
        self.paper_item_id = paper_item_id
        self.question = question
        self.prompt_text = prompt_text
        self.current_prompt_text = prompt_text
        self.language = language
        self.archive_graph = archive_graph
        self.graph = graph or build_text_followup_graph()

        self.turn_order: int = 1
        self.turns: List[Dict[str, Any]] = []
        self.current_transcript: str = ""

        # In-progress utterance state (Phase 3): _live_partial accumulates partial_transcript
        # deltas for whichever utterance is currently being spoken; it is folded into
        # current_transcript (and cleared) once that utterance's final_transcript arrives.
        self._live_partial: str = ""
        self._awaiting_final_transcript: bool = False
        self._final_transcript_event: asyncio.Event = asyncio.Event()

    @classmethod
    async def create_from_archive(
        cls,
        *,
        answer_id: str,
        exam_attempt_id: str,
        archive_graph: Any,
        graph=None,
    ) -> Optional["RealtimeExamSession"]:
        """Rebuilds a full session purely from the durable archive_graph
        checkpoint, keyed by answer_id -- used by AttemptConnection's
        `resume` handler so a reconnect (WebSocket drop, or the serving pod
        restarting on EKS) can restore an in-progress question WITHOUT the
        client resending question_start or any question data at all: the
        question snapshot (question/paper_item_id/language/prompt_text) was
        durably persisted once, at question_start, by
        turn_publisher.persist_question_snapshot; turn history/
        decision_reasons/the pending follow-up prompt are persisted per-turn
        by turn_publisher.publish_turn_if_new. Returns None if nothing was
        ever archived for this answer_id (nothing to resume -- callers
        should treat that as a resume that can't be honored, not a crash)."""
        resume_state = await turn_publisher.get_resume_state(archive_graph, answer_id)
        if resume_state is None:
            return None

        session = cls(
            answer_id=answer_id,
            exam_attempt_id=exam_attempt_id,
            paper_item_id=resume_state.get("paper_item_id"),
            question=resume_state.get("question"),
            prompt_text=resume_state.get("prompt_text"),
            language=resume_state.get("language") or "en-US",
            archive_graph=archive_graph,
            graph=graph,
        )
        session._apply_resume_state(resume_state)
        return session

    def on_speech_start(self) -> None:
        """A new utterance within this turn just started (VAD speech_start).
        Does not touch already-finalized text in current_transcript."""
        self._live_partial = ""

    def on_partial_transcript(self, text: Optional[str]) -> None:
        """Accumulate an in-progress utterance's transcription deltas.
        Best-effort only -- on_final_transcript is the authoritative text for
        this utterance once it arrives."""
        if text:
            self._live_partial += text

    def on_speech_end(self) -> None:
        """VAD detected the end of an utterance; its final_transcript is
        still in flight (transcription is slower than VAD). Callers that
        need the turn's full transcript right now should await
        wait_for_pending_transcript first."""
        self._awaiting_final_transcript = True
        self._final_transcript_event.clear()

    def on_final_transcript(self, text: Optional[str]) -> None:
        """Authoritative transcript for the utterance that just completed --
        appended to current_transcript (a turn may span multiple
        utterances), superseding that utterance's own partial deltas."""
        finalized = (text or self._live_partial or "").strip()
        if finalized:
            self.current_transcript = f"{self.current_transcript} {finalized}".strip()
        self._live_partial = ""
        self._awaiting_final_transcript = False
        self._final_transcript_event.set()

    async def wait_for_pending_transcript(self, timeout: float = PENDING_TRANSCRIPT_TIMEOUT_SECONDS) -> None:
        """Block briefly for a just-ended utterance's final_transcript before
        a turn_end reads current_transcript, so a turn_end arriving right on
        the heels of vad_speech_end doesn't race the (slightly slower)
        transcription. No-op if no utterance is currently pending."""
        if not self._awaiting_final_transcript:
            return
        try:
            await asyncio.wait_for(self._final_transcript_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "[session] timed out waiting for final_transcript, falling back to partial text: answer_id=%s",
                self.answer_id,
            )
            # Fall back to whatever partial text was accumulated rather than silently dropping
            # this utterance's words.
            if self._live_partial.strip():
                self.current_transcript = f"{self.current_transcript} {self._live_partial.strip()}".strip()
            self._live_partial = ""
            self._awaiting_final_transcript = False

    async def hydrate_from_archive(self) -> None:
        """Restore turn history (+ the pending follow-up prompt) from the
        durable archive_graph checkpoint if this answer_id already has
        archived turns -- called unconditionally by AttemptConnection on
        every question_start, including brand-new questions (where it's a
        cheap no-op: no checkpoint exists yet for this answer_id). This is
        what lets a question survive a reconnect where the client resends a
        fresh question_start (as opposed to a `resume`, which goes through
        create_from_archive instead) without losing progress, since
        self.turns/self.turn_order/self.current_prompt_text otherwise only
        ever lived in this process's memory. Does not restore
        self.current_transcript or the in-progress-utterance state -- a turn
        that was still being spoken when the connection dropped is genuinely
        lost and must be re-answered."""
        resume_state = await turn_publisher.get_resume_state(self.archive_graph, self.answer_id)
        if resume_state is None or not resume_state.get("turns"):
            return
        self._apply_resume_state(resume_state)

    def _apply_resume_state(self, resume_state: Dict[str, Any]) -> None:
        """Shared by hydrate_from_archive and create_from_archive: restores
        turn history and turn_order from resume_state["turns"], and
        current_prompt_text from resume_state["active_prompt_text"] -- the
        pending follow-up prompt persisted by
        turn_publisher.publish_turn_if_new after the LAST completed turn's
        decision (should_continue=True case). Falls back to the last
        archived turn's own prompt_text only if active_prompt_text was never
        persisted (e.g. archive predates this feature) -- that fallback is
        the PREVIOUS prompt, not the pending one, so it's a best-effort
        approximation, not a guarantee."""
        turns = resume_state.get("turns") or []
        if not turns:
            return

        self.turns = turns
        self.turn_order = max(int(t.get("turn_order") or 0) for t in turns) + 1

        pending_prompt = resume_state.get("active_prompt_text")
        if isinstance(pending_prompt, str) and pending_prompt.strip():
            self.current_prompt_text = pending_prompt.strip()
        else:
            last_prompt = turns[-1].get("prompt_text")
            if isinstance(last_prompt, str) and last_prompt.strip():
                self.current_prompt_text = last_prompt.strip()

        logger.info(
            "[session] resumed %d archived turn(s) for answer_id=%s, resuming at turn_order=%d",
            len(turns), self.answer_id, self.turn_order,
        )

    def _build_current_turn(self, transcript: str, word_count: int) -> Dict[str, Any]:
        return {
            "answer_id": self.answer_id,
            "paper_item_id": self.paper_item_id,
            "turn_order": self.turn_order,
            "turn_type": "MAIN" if self.turn_order == 1 else "FOLLOWUP",
            "prompt_text": self.current_prompt_text or self.prompt_text,
            "transcript": transcript,
            "word_count": word_count,
        }

    def decide_next_step(self, transcript: str, word_count: int) -> Dict[str, Any]:
        """Build FollowUpGraphState input directly from this session's held
        state and invoke the existing stateless text_followup_graph verbatim.
        Returns the decision dict immediately; schedules the durable
        archive/Kafka publish as a fire-and-forget background task rather
        than awaiting it, so Path A never blocks on Path B.
        """
        current_turn = self._build_current_turn(transcript, word_count)
        state = {
            "answer_id": self.answer_id,
            "exam_attempt_id": self.exam_attempt_id,
            "question": self.question,
            "turn_order": self.turn_order,
            "prompt_text": self.prompt_text,
            "active_prompt_text": self.current_prompt_text or self.prompt_text,
            "current_turn": current_turn,
            "turns": self.turns,
            "status": "idle",
        }

        result = self.graph.invoke(state)
        decision = result.get("decision") or {
            "should_continue": False,
            "next_prompt_text": None,
            "reason": result.get("error") or "decision_failed",
        }

        current_turn["decision_reason"] = decision.get("reason", "")
        # Record this turn locally so the next turn's prepare_turn_signals/
        # followup_decision sees correct cumulative history and turn_order.
        self.turns.append(current_turn)
        completed_turn_order = self.turn_order
        self.turn_order += 1
        next_active_prompt = decision.get("active_prompt_text")
        next_prompt_text = decision.get("next_prompt_text")
        if decision.get("should_continue"):
            if isinstance(next_active_prompt, str) and next_active_prompt.strip():
                self.current_prompt_text = next_active_prompt.strip()
            elif isinstance(next_prompt_text, str) and next_prompt_text.strip():
                self.current_prompt_text = next_prompt_text.strip()
        self.current_transcript = ""

        self.schedule_publish(completed_turn_order, reason=decision.get("reason", ""))

        return decision

    def schedule_publish(self, turn_order: int, reason: str = "") -> "asyncio.Task[None]":
        """Fire-and-forget: never awaited by decide_next_step's caller. A
        reference to the task is returned only so callers/tests that want to
        await completion explicitly may do so; nothing in the normal flow
        relies on awaiting it."""
        return asyncio.create_task(
            turn_publisher.publish_turn_if_new(
                self.archive_graph,
                self.answer_id,
                turn_order,
                reason=reason,
                exam_attempt_id=self.exam_attempt_id,
                active_prompt_text=self.current_prompt_text,
            )
        )
