"""Owns the active question session for one realtime attempt connection."""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from infra.database import archive_store
from infra.voice_live_client import VoiceLiveServerEvent
from node.state_models import QuestionContext
from realtime.question.session import QuestionSession
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


class QuestionSessionCoordinator:
    def __init__(
        self,
        *,
        exam_attempt_id: str,
        archive_graph: Any,
        text_followup_graph: Any,
        turn_processor: Optional[TurnProcessor] = None,
    ) -> None:
        self.exam_attempt_id = exam_attempt_id
        self.archive_graph = archive_graph
        self.text_followup_graph = text_followup_graph
        self.turn_processor = turn_processor or TurnProcessor()
        self.active_session: Optional[QuestionSession] = None

    async def start_question(self, message: Dict[str, Any]) -> QuestionStartResult:
        answer_id = message.get("answer_id")
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
        remaining_graded_seconds = message.get("remaining_graded_seconds")

        self.active_session = QuestionSession(
            answer_id=answer_id,
            exam_attempt_id=self.exam_attempt_id,
            paper_item_id=paper_item_id,
            question=question,
            prompt_text=prompt_text,
            language=language,
            remaining_graded_seconds=remaining_graded_seconds,
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
            remaining_graded_seconds=remaining_graded_seconds,
        )
        await archive_store.set_current_answer_id(
            self.archive_graph,
            self.exam_attempt_id,
            answer_id,
        )
        await self.active_session.hydrate_from_archive()

        logger.info(
            "[question_coordinator] started exam_attempt_id=%s answer_id=%s",
            self.exam_attempt_id,
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
                "[question_coordinator] turn_end without active session exam_attempt_id=%s",
                self.exam_attempt_id,
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
        if not math.isfinite(elapsed_seconds):
            return

        elapsed_seconds = max(0.0, elapsed_seconds)
        if elapsed_seconds <= session.speech_budget_elapsed_seconds:
            return

        session.speech_budget_elapsed_seconds = elapsed_seconds
        await archive_store.persist_speech_budget_elapsed(
            session.archive_graph,
            session.answer_id,
            elapsed_seconds,
        )

    async def resume(self, message: Dict[str, Any]) -> ResumeResult:
        answer_id = message.get("answer_id")
        session = await QuestionSession.create_from_archive(
            answer_id=answer_id,
            exam_attempt_id=self.exam_attempt_id,
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
                "[question_coordinator] resumed answer_id=%s next_turn=%d recovered_turn=%s",
                answer_id,
                session.turn_order,
                recovered.completed_turn_order if recovered else None,
            )
        else:
            logger.warning(
                "[question_coordinator] no archive to resume answer_id=%s",
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
        return ResumeResult(
            acknowledgement=acknowledgement,
            recovered_decision=recovered_decision,
            prompt_to_speak=prompt_to_speak,
        )

    def route_voice_event(self, event: VoiceLiveServerEvent) -> bool:
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
