from typing import List, Literal, Optional

from infra.message_broker.events.envelope import EventEnvelope
from schemas.common import _CamelMessage


class AnswerTurnPayload(_CamelMessage):
    answer_id: Optional[str] = None
    turn_order: int
    turn_type: Optional[str] = None
    prompt_text: Optional[str] = None
    audio_url: Optional[str] = None
    transcript: str = ""
    duration_seconds: Optional[int] = None
    word_count: Optional[int] = None
    answered_at: Optional[str] = None


class AnswerTurnsRecordedPayload(_CamelMessage):
    turns: List[AnswerTurnPayload]
    reason: str = ""


class AnswerTurnsRecordedEvent(EventEnvelope):
    event_type: Literal["AnswerTurnsRecorded"] = "AnswerTurnsRecorded"

    answer_id: str
    payload: AnswerTurnsRecordedPayload
