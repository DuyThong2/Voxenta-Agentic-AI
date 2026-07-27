from typing import Literal, Optional

from events.envelope import EventEnvelope
from schemas.common import _CamelMessage


class ExamAttemptForceEndRequestedPayload(_CamelMessage):
    reason: Optional[str] = None


class ExamAttemptForceEndRequestedEvent(EventEnvelope):
    event_type: Literal["ExamAttemptForceEndRequested"] = "ExamAttemptForceEndRequested"

    exam_attempt_id: str
    payload: ExamAttemptForceEndRequestedPayload
