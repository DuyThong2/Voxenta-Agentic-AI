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
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from node.followUpDecisionGraph.graphConfig import build_text_followup_graph
from node.state_models import QuestionContext
from realtime import turn_publisher

logger = logging.getLogger(__name__)


class RealtimeExamSession:
    """Holds turn-accumulation state for exactly one question
    (answer_id). turn_order starts at 1 and increments per turn within this
    session; a new question always gets a brand-new RealtimeExamSession with
    turn_order reset to 1, created by AttemptConnection."""

    def __init__(
        self,
        *,
        answer_id: str,
        question: Optional[QuestionContext],
        prompt_text: Optional[str],
        language: str,
        archive_graph: Any,
        graph=None,
    ) -> None:
        self.answer_id = answer_id
        self.question = question
        self.prompt_text = prompt_text
        self.language = language
        self.archive_graph = archive_graph
        self.graph = graph or build_text_followup_graph()

        self.turn_order: int = 1
        self.turns: List[Dict[str, Any]] = []
        self.current_transcript: str = ""

    def append_transcript_chunk(self, text: str) -> None:
        """Accumulate live transcript text for the turn currently in
        progress. Mirrors Phase 3's eventual VAD-driven accumulation, but
        here it's just string concatenation off the placeholder text
        protocol."""
        if not text:
            return
        self.current_transcript = f"{self.current_transcript} {text}".strip()

    def _build_current_turn(self, transcript: str, word_count: int) -> Dict[str, Any]:
        return {
            "answer_id": self.answer_id,
            "turn_order": self.turn_order,
            "turn_type": "MAIN" if self.turn_order == 1 else "FOLLOWUP",
            "prompt_text": self.prompt_text,
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
            "question": self.question,
            "turn_order": self.turn_order,
            "prompt_text": self.prompt_text,
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

        # Record this turn locally so the next turn's prepare_turn_signals/
        # followup_decision sees correct cumulative history and turn_order.
        self.turns.append(current_turn)
        completed_turn_order = self.turn_order
        self.turn_order += 1
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
                self.archive_graph, self.answer_id, turn_order, reason=reason,
            )
        )
