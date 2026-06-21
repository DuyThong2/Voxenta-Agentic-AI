from typing import List, Literal, Optional

from infra.message_broker.events.envelope import EventEnvelope
from infra.message_broker.events.shared import TurnInput
from pydantic import Field
from schemas.framework import CriterionFramework
from schemas.common import _CamelMessage
from schemas.evaluation_event import EvaluationGuideInput


class ExamAttemptEvaluationRequestedPayload(_CamelMessage):
    question_text: Optional[str] = None
    question_type: Optional[str] = None
    difficulty_level: Optional[str] = None
    duration_seconds: Optional[int] = None
    min_response_seconds: Optional[int] = None
    max_response_seconds: Optional[int] = None
    topic_name: Optional[str] = None
    topic_description: Optional[str] = None
    evaluation_guide: Optional[EvaluationGuideInput] = None
    mode: str = "unscripted"
    reference_text: Optional[str] = None
    language: str = "en-US"
    criteria_frameworks: List[CriterionFramework] = Field(default_factory=list)
    turns: List[TurnInput]


class ExamAttemptEvaluationRequestedEvent(EventEnvelope):
    event_type: Literal["ExamAttemptEvaluationRequested"] = "ExamAttemptEvaluationRequested"

    exam_attempt_id: str
    answer_id: str
    question_id: str
    payload: ExamAttemptEvaluationRequestedPayload
