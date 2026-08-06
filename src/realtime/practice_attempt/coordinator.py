"""Owns the active question session for one practice realtime connection --
parallel to realtime/question/coordinator.py (gói 11 mục 2.3/2.5), reusing TurnProcessor
as-is via composition instead of subclassing/branching the exam coordinator.

The one new responsibility exam's QuestionSessionCoordinator doesn't have:
resolve_and_push_next_question -- called when a question's follow-up chain ends
(should_continue == False), it asks Java for the next MAIN question (mục 2.2 bước 3)
and builds a new PracticeQuestionSession from that response, instead of waiting for the
client to send another question_start the way exam always does.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from infra import practice_session_client
from infra.database import archive_store
from node.state_models import QuestionContext
from realtime.practice_attempt.session import PracticeQuestionSession
from realtime.turn.processor import TurnProcessingResult, TurnProcessor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuestionStartResult:
    answer_id: str
    section_instruction: str


@dataclass(frozen=True)
class ResumeResult:
    acknowledgement: Dict[str, Any]
    recovered_decision: Optional[Dict[str, Any]]
    prompt_to_speak: Optional[str]


@dataclass(frozen=True)
class NextQuestionPushResult:
    """What PracticeAttemptConnection needs to push a next_question message, or None
    fields when the session ended (no_more_questions)."""
    ended: bool
    reason: Optional[str]
    answer_id: Optional[str]
    question: Optional[Dict[str, Any]]
    prompt_text: Optional[str]
    # "Da noi / ngan sach" cho thanh tien do tren may hoc sinh. Java gui kem moi lan resolve
    # cau hoi vi luot dau chua nop nen chua co ket qua nop luot nao de lay ngan sach tu do.
    # None khi chinh cuoc goi Java that bai (khong bia so 0 -- client se an thanh tien do).
    session_spoken_seconds: Optional[int] = None
    session_budget_seconds: Optional[int] = None


class PracticeQuestionSessionCoordinator:
    def __init__(
        self,
        *,
        practice_session_id: str,
        archive_graph: Any,
        text_followup_graph: Any,
        turn_processor: Optional[TurnProcessor] = None,
    ) -> None:
        self.practice_session_id = practice_session_id
        self.archive_graph = archive_graph
        self.text_followup_graph = text_followup_graph
        self.turn_processor = turn_processor or TurnProcessor()
        self.active_session: Optional[PracticeQuestionSession] = None

    def _answer_id_for(self, question_id: str) -> str:
        return f"{self.practice_session_id}:{question_id}"

    async def start_question(self, message: Dict[str, Any]) -> QuestionStartResult:
        question_id = message.get("question_id")
        answer_id = self._answer_id_for(question_id)
        paper_item_id = message.get("paper_item_id")
        question_payload = message.get("question") or {}
        question = (
            QuestionContext.model_validate(question_payload)
            if question_payload
            else None
        )
        prompt_text = (
            question_payload.get("question_text")
            if isinstance(question_payload, dict)
            else None
        )
        language = message.get("language", "en-US")

        self.active_session = PracticeQuestionSession(
            answer_id=answer_id,
            practice_session_id=self.practice_session_id,
            question_id=str(question_id),
            paper_item_id=paper_item_id,
            question=question,
            prompt_text=prompt_text,
            language=language,
            archive_graph=self.archive_graph,
            graph=self.text_followup_graph,
        )
        await archive_store.persist_question_snapshot(
            self.archive_graph,
            answer_id,
            question=question,
            paper_item_id=paper_item_id,
            language=language,
            prompt_text=prompt_text,
        )
        await archive_store.set_current_answer_id(
            self.archive_graph,
            self.practice_session_id,
            answer_id,
        )
        await self.active_session.hydrate_from_archive()

        logger.info(
            "[practice_question_coordinator] started practice_session_id=%s answer_id=%s",
            self.practice_session_id,
            answer_id,
        )
        return QuestionStartResult(
            answer_id=answer_id,
            section_instruction=str(message.get("section_instruction") or "").strip(),
        )

    def present_question(self, message: Dict[str, Any]) -> Optional[str]:
        prompt_text = message.get("prompt_text")
        if self.active_session is not None and prompt_text:
            self.active_session.current_prompt_text = prompt_text
        return prompt_text

    async def process_turn(
        self,
        message: Dict[str, Any],
    ) -> Optional[TurnProcessingResult]:
        if self.active_session is None:
            logger.warning(
                "[practice_question_coordinator] turn_end without active session practice_session_id=%s",
                self.practice_session_id,
            )
            return None
        return await self.turn_processor.process(self.active_session, message)

    async def checkpoint_speech_budget(self, message: Dict[str, Any]) -> None:
        session = self.active_session
        if session is None or message.get("answer_id") != session.answer_id:
            return
        try:
            elapsed_seconds = float(message.get("elapsed_seconds"))
        except (TypeError, ValueError):
            return
        if elapsed_seconds <= session.speech_budget_elapsed_seconds:
            return
        session.speech_budget_elapsed_seconds = elapsed_seconds
        await archive_store.persist_speech_budget_elapsed(
            session.archive_graph,
            session.answer_id,
            elapsed_seconds,
        )

    async def resume(self, message: Dict[str, Any]) -> ResumeResult:
        question_id = message.get("question_id")
        answer_id = self._answer_id_for(question_id) if question_id else message.get("answer_id")
        session = await PracticeQuestionSession.create_from_archive(
            answer_id=answer_id,
            practice_session_id=self.practice_session_id,
            archive_graph=self.archive_graph,
            graph=self.text_followup_graph,
        )

        last_archived_turn_order = 0
        recovered: Optional[TurnProcessingResult] = None
        if session is not None:
            self.active_session = session
            last_archived_turn_order = session.turn_order - 1
            recovered = await self.turn_processor.recover_pending(session)
            logger.info(
                "[practice_question_coordinator] resumed answer_id=%s next_turn=%d recovered_turn=%s",
                answer_id,
                session.turn_order,
                recovered.completed_turn_order if recovered else None,
            )
        else:
            logger.warning(
                "[practice_question_coordinator] no archive to resume answer_id=%s",
                answer_id,
            )

        acknowledgement: Dict[str, Any] = {
            "type": "resume_ack",
            "answer_id": answer_id,
            "last_archived_turn_order": last_archived_turn_order,
        }
        recovered_decision = recovered.decision if recovered else None
        if recovered is not None:
            acknowledgement["recovered_turn_order"] = recovered.completed_turn_order
            acknowledgement["decision"] = recovered.decision

        prompt_to_speak = (
            session.current_prompt_text
            if recovered is None and session is not None
            else None
        )
        if prompt_to_speak:
            # Was computed but never actually reached the client before -- carry it in the
            # ack itself so Flutter can re-send present_question with it (see
            # PracticeAttemptConnection._handle_resume and practice_session_screen.dart's
            # _handleResumeAck), the same "client always re-presents, server never
            # auto-speaks" rule as every other transition.
            acknowledgement["prompt_to_speak"] = prompt_to_speak
        return ResumeResult(
            acknowledgement=acknowledgement,
            recovered_decision=recovered_decision,
            prompt_to_speak=prompt_to_speak,
        )

    async def resolve_and_push_next_question(self) -> NextQuestionPushResult:
        """Called when the current question's follow-up chain ends
        (should_continue == False). Asks Java for the next MAIN question and, if one
        comes back, builds a brand new PracticeQuestionSession for it -- there is no
        client question_start for MAIN-question transitions, unlike exam."""
        try:
            data = await practice_session_client.request_next_question(self.practice_session_id)
        except (httpx.HTTPError, KeyError):
            logger.exception(
                "[practice_question_coordinator] next-question call failed practice_session_id=%s",
                self.practice_session_id,
            )
            return NextQuestionPushResult(
                ended=True, reason="next_question_call_failed",
                answer_id=None, question=None, prompt_text=None,
            )

        if data.get("status") != "ok":
            return NextQuestionPushResult(
                ended=True, reason=data.get("reason") or "no_more_questions",
                answer_id=None, question=None, prompt_text=None,
                session_spoken_seconds=data.get("sessionSpokenSeconds"),
                session_budget_seconds=data.get("sessionBudgetSeconds"),
            )

        question_payload = data["question"]
        question_id = str(question_payload["questionId"])
        answer_id = self._answer_id_for(question_id)
        prompt_text = question_payload.get("questionText")

        # min_response_seconds la MOC de SignalNode biet tra loi da du chua. Truoc day khong
        # gui, nen _resolve_target_response_seconds roi xuong lay TRAN lam moc: mot cau tra loi
        # tron ven 18 giay tren tran 45 giay bi doc thanh "moi dat 0.4", va nguong
        # followup_pressure=high (ratio >= 1.15) doi noi 52 giay khi tran la 45 -- khong bao gio
        # voi toi. Tin hieu thoi gian vi the bi vo hieu, chi con so luot dieu khien viec dung.
        question_context_payload = {
            "question_text": question_payload.get("questionText"),
            "duration_seconds": question_payload.get("maxResponseSeconds"),
            "min_response_seconds": question_payload.get("minResponseSeconds"),
            "max_response_seconds": question_payload.get("maxResponseSeconds"),
        }
        question = QuestionContext.model_validate(question_context_payload)

        self.active_session = PracticeQuestionSession(
            answer_id=answer_id,
            practice_session_id=self.practice_session_id,
            question_id=question_id,
            paper_item_id=str(question_payload.get("slot")),
            question=question,
            prompt_text=prompt_text,
            language="en-US",
            archive_graph=self.archive_graph,
            graph=self.text_followup_graph,
            last_question=bool(question_payload.get("lastQuestion")),
        )
        await archive_store.persist_question_snapshot(
            self.archive_graph,
            answer_id,
            question=question,
            paper_item_id=question_payload.get("slot"),
            language="en-US",
            prompt_text=prompt_text,
        )
        await archive_store.set_current_answer_id(
            self.archive_graph,
            self.practice_session_id,
            answer_id,
        )

        logger.info(
            "[practice_question_coordinator] resolved next question practice_session_id=%s answer_id=%s",
            self.practice_session_id,
            answer_id,
        )
        return NextQuestionPushResult(
            ended=False, reason=None,
            answer_id=answer_id, question=question_payload, prompt_text=prompt_text,
            session_spoken_seconds=data.get("sessionSpokenSeconds"),
            session_budget_seconds=data.get("sessionBudgetSeconds"),
        )

    def route_voice_event(self, event) -> bool:
        session = self.active_session
        if session is None:
            return False

        accumulator = session.transcript
        if event.kind == "vad_speech_start":
            accumulator.on_speech_start()
        elif event.kind == "partial_transcript":
            accumulator.on_partial_transcript(event.text)
        elif event.kind == "vad_speech_end":
            accumulator.on_speech_end()
        elif event.kind == "final_transcript":
            accumulator.on_final_transcript(event.text, event.confidence)
        else:
            return False
        return True

    def clear(self) -> None:
        self.active_session = None
