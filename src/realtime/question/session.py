"""Question-scoped realtime state.

The session owns question identity, prompt history and turn order. Transcript
accumulation and turn processing are delegated to their turn-scoped components.
"""

import logging
from typing import Any, Dict, List, Optional

from infra.database import archive_store
from node.followUpDecisionGraph.graphConfig import build_text_followup_graph
from node.state_models import QuestionContext
from realtime.turn.transcript_accumulator import TranscriptAccumulator

logger = logging.getLogger(__name__)


class QuestionSession:
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
        self.turn_order = 1
        self.turns: List[Dict[str, Any]] = []
        self.transcript = TranscriptAccumulator()

    @classmethod
    async def create_from_archive(
        cls,
        *,
        answer_id: str,
        exam_attempt_id: str,
        archive_graph: Any,
        graph=None,
    ) -> Optional["QuestionSession"]:
        resume_state = await archive_store.get_resume_state(archive_graph, answer_id)
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

    async def hydrate_from_archive(self) -> None:
        resume_state = await archive_store.get_resume_state(
            self.archive_graph,
            self.answer_id,
        )
        if resume_state is None or not resume_state.get("turns"):
            return
        self._apply_resume_state(resume_state)

    def _apply_resume_state(self, resume_state: Dict[str, Any]) -> None:
        turns = resume_state.get("turns") or []
        if not turns:
            return

        self.turns = turns
        self.turn_order = max(int(turn.get("turn_order") or 0) for turn in turns) + 1

        pending_prompt = resume_state.get("active_prompt_text")
        if isinstance(pending_prompt, str) and pending_prompt.strip():
            self.current_prompt_text = pending_prompt.strip()
        else:
            last_prompt = turns[-1].get("prompt_text")
            if isinstance(last_prompt, str) and last_prompt.strip():
                self.current_prompt_text = last_prompt.strip()

        logger.info(
            "[question_session] resumed %d turn(s) answer_id=%s next_turn=%d",
            len(turns),
            self.answer_id,
            self.turn_order,
        )

    def build_current_turn(
        self,
        transcript: str,
        word_count: int,
        duration_seconds: Optional[float],
    ) -> Dict[str, Any]:
        return {
            "answer_id": self.answer_id,
            "paper_item_id": self.paper_item_id,
            "turn_order": self.turn_order,
            "turn_type": "MAIN" if self.turn_order == 1 else "FOLLOWUP",
            "prompt_text": self.current_prompt_text or self.prompt_text,
            "transcript": transcript,
            "word_count": word_count,
            "duration_seconds": duration_seconds,
        }

    def build_decision_state(self, current_turn: Dict[str, Any]) -> Dict[str, Any]:
        return {
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

    def complete_turn(
        self,
        current_turn: Dict[str, Any],
        decision: Dict[str, Any],
    ) -> int:
        completed_turn_order = self.turn_order
        current_turn["decision_reason"] = decision.get("reason", "")
        self.turns.append(current_turn)
        self.turn_order += 1

        if decision.get("should_continue"):
            active_prompt = decision.get("active_prompt_text")
            next_prompt = decision.get("next_prompt_text")
            if isinstance(active_prompt, str) and active_prompt.strip():
                self.current_prompt_text = active_prompt.strip()
            elif isinstance(next_prompt, str) and next_prompt.strip():
                self.current_prompt_text = next_prompt.strip()

        return completed_turn_order
