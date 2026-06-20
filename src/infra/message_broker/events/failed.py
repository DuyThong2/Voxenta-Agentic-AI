from typing import Literal

from infra.message_broker.events.envelope import EventEnvelope
from schemas.common import _CamelMessage


class ExamAttemptEvaluationFailedPayload(_CamelMessage):
    error: str
    retry_count: int


class ExamAttemptEvaluationFailedEvent(EventEnvelope):
    """Published to the completed topic instead of a separate dead-letter topic."""

    event_type: Literal["ExamAttemptEvaluationFailed"] = "ExamAttemptEvaluationFailed"

    exam_attempt_id: str
    answer_id: str
    question_id: str
    payload: ExamAttemptEvaluationFailedPayload
