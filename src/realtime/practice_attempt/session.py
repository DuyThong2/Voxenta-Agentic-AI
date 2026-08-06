"""Question-scoped realtime state for PRACTICE -- parallel to
realtime/question/session.py, not a shared base class (gói 11 mục 2.3: "tạo class song
song riêng... không thêm nhánh if practice else exam vào code dùng chung").

Reuses infra.database.archive_store as-is: its functions are already generic over the
thread_id string they're keyed by (see archive_store.archive_config), so no new archive
module is needed -- practice just passes its own id values (see coordinator.py for how
those ids are built) instead of an exam_attempt_id/answer_id.
"""

import logging
import math
from typing import Any, Dict, List, Optional

from infra.database import archive_store
from node.followUpDecisionGraph.graphConfig import build_text_followup_graph
from node.state_models import QuestionContext
from realtime.turn.transcript_accumulator import TranscriptAccumulator

logger = logging.getLogger(__name__)


class PracticeQuestionSession:
    def __init__(
        self,
        *,
        answer_id: str,
        practice_session_id: str,
        question_id: str,
        paper_item_id: Optional[str],
        question: Optional[QuestionContext],
        prompt_text: Optional[str],
        language: str,
        archive_graph: Any,
        graph=None,
        last_question: bool = False,
    ) -> None:
        self.answer_id = answer_id
        self.practice_session_id = practice_session_id
        # The real Java new_practice_question id -- distinct from answer_id (a
        # session:question composite key used only for archive/resume, see coordinator.py).
        # PracticeAttemptConnection needs this to call /internal/practice-sessions/{id}/turns.
        self.question_id = question_id
        self.paper_item_id = paper_item_id
        self.question = question
        self.prompt_text = prompt_text
        self.current_prompt_text = prompt_text
        self.language = language
        self.archive_graph = archive_graph
        self.graph = graph or build_text_followup_graph()
        self.turn_order = 1
        self.turns: List[Dict[str, Any]] = []
        self.speech_budget_elapsed_seconds = 0.0
        # Java bao day la cau CUOI: ngan sach con lai khong du cho mot cau co binh
        # thuong nen no da duoc may do vua dung phan con lai. Tra loi xong thi dong phien,
        # khong hoi tiep -- hoi tiep chi de nhan lai budget_exhausted, ton mot vong goi.
        self.last_question = last_question
        self.transcript = TranscriptAccumulator()

    @classmethod
    async def create_from_archive(
        cls,
        *,
        answer_id: str,
        practice_session_id: str,
        archive_graph: Any,
        graph=None,
    ) -> Optional["PracticeQuestionSession"]:
        resume_state = await archive_store.get_resume_state(archive_graph, answer_id)
        if resume_state is None:
            return None

        # answer_id is always f"{practice_session_id}:{question_id}" (see
        # coordinator.py._answer_id_for) -- both halves are UUIDs, never contain ":",
        # so splitting on the first ":" recovers question_id without a second persisted field.
        question_id = answer_id.split(":", 1)[1] if ":" in answer_id else answer_id
        session = cls(
            answer_id=answer_id,
            practice_session_id=practice_session_id,
            question_id=question_id,
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
        elapsed_seconds = resume_state.get("speech_budget_elapsed_seconds")
        if isinstance(elapsed_seconds, (int, float)) and math.isfinite(elapsed_seconds):
            self.speech_budget_elapsed_seconds = max(
                self.speech_budget_elapsed_seconds,
                float(elapsed_seconds),
            )

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
            "[practice_question_session] resumed %d turn(s) answer_id=%s next_turn=%d",
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
            "exam_attempt_id": self.practice_session_id,
            "question": self.question,
            "turn_order": self.turn_order,
            "prompt_text": self.prompt_text,
            "active_prompt_text": self.current_prompt_text or self.prompt_text,
            "remaining_graded_seconds": None,
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
