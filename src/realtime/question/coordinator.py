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
        # Moc dong ho luc cau nay bat dau -- xem archive_store.persist_question_snapshot.
        remaining_seconds_at_question_start = message.get(
            "remaining_seconds_at_question_start"
        )

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
            remaining_seconds_at_question_start=remaining_seconds_at_question_start,
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

        # CHỈ GHI LOG, KHÔNG CHẶN -- cột mốc để bắt tận tay việc resume kéo phiên về câu cũ.
        #
        # Đo thật 2026-08-26, ca 01a03cb8: thí sinh đứt mạng ở câu 2, nối lại, rồi trả lời câu 2 --
        # nhưng bản ghi lại nằm ở câu 1:
        #     paper_item 01a03c55  lượt 1 MAIN     "How important is entertainment..."  -> giải trí
        #     paper_item 01a03c55  lượt 2 FOLLOWUP "Take 25 seconds to think..."        -> tả bức ảnh
        # Cùng một paper_item_id. Câu trả lời của câu 2 thành lượt follow-up của câu 1, còn câu 2
        # thì rỗng. Hỏng cả hai câu, và hỏng im lặng -- không lỗi, không log, chỉ sai bản ghi.
        #
        # GỐC nằm ở WPF, không phải ở đây: QuestionFlowRunner chỉ gọi SetResumeCheckpoint ở nhánh
        # vào-lại và sau khi một lượt có quyết định, KHÔNG gọi khi bắt đầu câu mới. Nên từ lúc sang
        # câu mới tới lúc xong lượt đầu, checkpoint vẫn trỏ câu trước, và `resume` khai sai câu.
        # Bản vá thật là thêm một lời gọi ở nhánh câu mới bên đó.
        #
        # VÌ SAO CHỈ GHI LOG MÀ KHÔNG CHẶN -- đã cân nhắc chặn rồi bỏ:
        # `start_question` là thứ ĐẶT `current_answer_id`. Nên trong khe giữa lúc client đặt
        # checkpoint câu mới và lúc `question_start` tới được server, client đã trỏ câu 2 trong khi
        # `current_answer_id` còn là câu 1. Chặn ở đó là từ chối oan một resume ĐÚNG, giữ phiên lại
        # câu cũ -- tự tay dựng lại đúng lỗi đang chữa. Ghi log thì có bằng chứng mà không thêm
        # đường hỏng nào.
        #
        # Ba giá trị dưới đây là đủ để lần ra bên nào sai: client khai gì, phiên đang giữ gì, và
        # trạng thái bền ghi gì.
        current_answer_id = await archive_store.get_current_answer_id(
            self.archive_graph,
            self.exam_attempt_id,
        )
        if answer_id and current_answer_id and answer_id != current_answer_id:
            logger.warning(
                "[question_coordinator] RESUME LECH CAU exam_attempt_id=%s client_khai=%s "
                "phien_dang_giu=%s trang_thai_ben=%s -- van cho qua, xem chu thich",
                self.exam_attempt_id,
                answer_id,
                self.active_session.answer_id if self.active_session else None,
                current_answer_id,
            )

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
