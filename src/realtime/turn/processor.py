"""Turn-end processing shared by the live path and reconnect recovery."""

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from infra.database import archive_store
from infra.message_broker.publishers import turn_publisher
from realtime.question.session import QuestionSession
from realtime.background import spawn

logger = logging.getLogger(__name__)

_CLARIFICATION_REASONS = {
    "clarify_prompt",
    "decline_repair",
    "remind_respectfully",
}


MAX_CONSECUTIVE_CLARIFICATIONS = 3
"""Bao nhieu lan AI duoc hoi lai LIEN TIEP truoc khi bo cuoc va chuyen cau.

3 la du de xu ly nhieu am thanh nhat thoi (mic tre, tieng on, hoc sinh ngap ngung) ma khong de
mot phien im lang keo dai vo tan. Vuot tran thi ket thuc cau voi ly do
`client_max_clarifications_reached` -- khac han `client_max_turns_reached` de con truy duoc
nguyen nhan tu log.
"""


def _trailing_clarifications(session) -> int:
    """Dem so luot HOI LAI o CUOI chuoi. Mot luot noi that o giua se reset ve 0."""
    count = 0
    for turn in reversed(getattr(session, "turns", None) or []):
        if not is_clarification_reason(turn.get("decision_reason")):
            break
        count += 1
    return count


def is_clarification_reason(reason: Optional[str]) -> bool:
    return bool(reason) and (
        str(reason).lower().startswith("clarification_")
        or str(reason).lower() in _CLARIFICATION_REASONS
    )


def _optional_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _optional_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class TurnLimitPolicy:
    legacy_last_allowed: bool
    speech_budget_exceeded: bool
    assessment_turn_count: Optional[int]
    max_assessment_turns: Optional[int]

    @classmethod
    def from_mapping(cls, value: Dict[str, Any]) -> "TurnLimitPolicy":
        return cls(
            legacy_last_allowed=bool(value.get("is_last_allowed_turn")),
            speech_budget_exceeded=bool(value.get("speech_budget_exceeded")),
            assessment_turn_count=_optional_int(value.get("assessment_turn_count")),
            max_assessment_turns=_optional_int(value.get("max_assessment_turns")),
        )

    @property
    def has_structured_turn_limit(self) -> bool:
        return (
            self.assessment_turn_count is not None
            and self.max_assessment_turns is not None
            and self.max_assessment_turns > 0
        )


@dataclass(frozen=True)
class TurnProcessingResult:
    decision: Dict[str, Any]
    completed_turn_order: int


class TurnProcessor:
    async def process(
        self,
        session: QuestionSession,
        message: Dict[str, Any],
    ) -> TurnProcessingResult:
        explicit_text = message.get("text")
        if explicit_text is not None:
            transcript = str(explicit_text)
            transcript_confidence = None
            session.transcript.reset()
        else:
            snapshot = await session.transcript.consume_turn()
            transcript = snapshot.text
            transcript_confidence = snapshot.confidence

        duration_seconds = _optional_float(message.get("duration_seconds"))
        policy = TurnLimitPolicy.from_mapping(message)

        # TurnProcessor duoc dung chung cho CA hai loai phien, ma chung dat ten id khac nhau:
        # phien thi co exam_attempt_id, phien luyen co practice_session_id. Truoc day dong log
        # nay goi thang session.exam_attempt_id nen moi luot noi cua phien LUYEN deu chet bang
        # AttributeError -- mot cau log lam hong ca lenh xu ly luot, du no khong tham gia gi
        # vao nghiep vu. Lay id nao co, khong gia dinh loai phien.
        logger.info(
            "[realtime_transcript] session_id=%s answer_id=%s turn_order=%d text=%r",
            getattr(session, "exam_attempt_id", None)
            or getattr(session, "practice_session_id", None),
            session.answer_id,
            session.turn_order,
            transcript,
        )

        # Recovery requires this marker to exist before the potentially slow LLM call.
        await archive_store.persist_realtime_transcript(
            session.archive_graph,
            session.answer_id,
            session.turn_order,
            transcript,
            is_last_allowed_turn=policy.legacy_last_allowed,
            speech_budget_exceeded=policy.speech_budget_exceeded,
            assessment_turn_count=policy.assessment_turn_count,
            max_assessment_turns=policy.max_assessment_turns,
            duration_seconds=duration_seconds,
            confidence=transcript_confidence,
        )

        word_count = _optional_int(message.get("word_count"))
        if word_count is None:
            word_count = len(transcript.split()) if transcript else 0

        return await self._decide_and_complete(
            session,
            transcript,
            word_count,
            duration_seconds,
            policy,
        )

    async def recover_pending(
        self,
        session: QuestionSession,
    ) -> Optional[TurnProcessingResult]:
        pending_turn_order = session.turn_order
        resume_state = await archive_store.get_resume_state(
            session.archive_graph,
            session.answer_id,
        )
        pending_entry = None
        if resume_state is not None:
            transcripts = resume_state.get("realtime_transcripts") or []
            pending_entry = next(
                (
                    entry
                    for entry in transcripts
                    if entry.get("turn_order") == pending_turn_order
                ),
                None,
            )

        if pending_entry is not None:
            transcript = pending_entry.get("text") or ""
            duration_seconds = _optional_float(pending_entry.get("duration_seconds"))
            logger.info(
                "[turn_processor] recovering pending decision answer_id=%s turn_order=%d",
                session.answer_id,
                pending_turn_order,
            )
            return await self._decide_and_complete(
                session,
                transcript,
                len(transcript.split()) if transcript else 0,
                duration_seconds,
                TurnLimitPolicy.from_mapping(pending_entry),
            )

        if not session.turns:
            return None

        last_turn = session.turns[-1]
        last_turn_order = _optional_int(last_turn.get("turn_order"))
        if last_turn_order is None:
            return None

        return TurnProcessingResult(
            decision={
                "should_continue": bool(last_turn.get("should_continue")),
                "next_prompt_text": last_turn.get("next_prompt_text"),
                "reason": last_turn.get("decision_reason", ""),
            },
            completed_turn_order=last_turn_order,
        )

    async def _decide_and_complete(
        self,
        session: QuestionSession,
        transcript: str,
        word_count: int,
        duration_seconds: Optional[float],
        policy: TurnLimitPolicy,
    ) -> TurnProcessingResult:
        current_turn = session.build_current_turn(
            transcript,
            word_count,
            duration_seconds,
        )
        state = session.build_decision_state(current_turn)
        result = await asyncio.to_thread(session.graph.invoke, state)
        decision = result.get("decision") or {
            "should_continue": False,
            "next_prompt_text": None,
            "reason": result.get("error") or "decision_failed",
        }
        decision = self._apply_turn_limit(
            decision, policy, _trailing_clarifications(session)
        )
        completed_turn_order = session.complete_turn(current_turn, decision)

        # CHỈ phiên thi mới publish. Sự kiện này là AnswerTurnsRecordedEvent -- lượt nói của
        # BÀI THI; phiên luyện có đường riêng (`practice_session_client.submit_turn` gọi thẳng
        # HTTP sang Java) nên bắn thêm vào topic thi là gửi rác cho consumer chấm thi.
        #
        # Truyền exam_attempt_id=None cũng KHÔNG đúng: sự kiện vẫn được publish, chỉ là thiếu
        # session_id. Điều kiện phải là "có phải phiên thi không", không phải "id có null không".
        exam_attempt_id = getattr(session, "exam_attempt_id", None)
        if exam_attempt_id is not None:
            spawn(
                turn_publisher.publish_turn_if_new(
                    session.archive_graph,
                    session.answer_id,
                    completed_turn_order,
                    reason=decision.get("reason", ""),
                    exam_attempt_id=exam_attempt_id,
                    active_prompt_text=session.current_prompt_text,
                    should_continue=bool(decision.get("should_continue")),
                    next_prompt_text=decision.get("next_prompt_text"),
                )
            )
        return TurnProcessingResult(decision, completed_turn_order)

    @staticmethod
    def _apply_turn_limit(
        decision: Dict[str, Any],
        policy: TurnLimitPolicy,
        trailing_clarifications: int = 0,
    ) -> Dict[str, Any]:
        if not decision.get("should_continue"):
            return decision

        # Tran rieng cho luot HOI LAI lien tiep.
        #
        # Hai phanh ben duoi deu vo hieu khi hoc sinh im lang: luot hoi lai bi loai khoi
        # max_assessment_turns (co y, de "em noi lai giup co" khong an mat mot luot cham), con
        # speech_budget tinh theo GIAY NOI nen im lang khong lam no tang. Khong co tran nay thi
        # im lang -> hoi lai -> im lang -> lap mai: phien khong tu ket thuc, quota khong tru duoc
        # vi luot 0 giay, nhung moi vong van goi LLM that.
        #
        # Dem theo CHUOI CUOI chu khong phai tong: mot luot noi that o giua reset bo dem, dung
        # y nghia "AI hoi lai lien tiep bao nhieu lan ma van khong nghe duoc gi".
        if (
            is_clarification_reason(decision.get("reason"))
            and trailing_clarifications + 1 >= MAX_CONSECUTIVE_CLARIFICATIONS
        ):
            return {
                **decision,
                "should_continue": False,
                "next_prompt_text": None,
                "reason": "client_max_clarifications_reached",
            }

        if policy.speech_budget_exceeded:
            return {
                **decision,
                "should_continue": False,
                "next_prompt_text": None,
                "reason": "client_speech_budget_exceeded",
            }

        if policy.has_structured_turn_limit:
            current_is_assessment = not is_clarification_reason(decision.get("reason"))
            reaches_limit = (
                current_is_assessment
                and policy.assessment_turn_count + 1 >= policy.max_assessment_turns
            )
            if reaches_limit:
                return {
                    **decision,
                    "should_continue": False,
                    "next_prompt_text": None,
                    "reason": "client_max_turns_reached",
                }
            return decision

        if policy.legacy_last_allowed:
            return {
                **decision,
                "should_continue": False,
                "next_prompt_text": None,
                "reason": "client_max_turns_reached",
            }
        return decision
