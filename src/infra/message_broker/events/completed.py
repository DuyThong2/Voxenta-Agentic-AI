from typing import Any, Dict, List, Literal

from pydantic import Field

from infra.message_broker.events.envelope import EventEnvelope
from infra.message_broker.events.shared import EvaluationSignals, TurnDetail
from schemas.common import _CamelMessage
from schemas.scoring import CriterionScore
from schemas.validity import ValidityResult


class ExamAttemptEvaluationCompletedPayload(_CamelMessage):
    turns: List[TurnDetail] = Field(default_factory=list)
    criteria: Dict[str, CriterionScore] = Field(default_factory=dict)
    signals: EvaluationSignals
    validity: ValidityResult

    feedback_summary: str = ""
    suggestions: List[Dict[str, Any]] = Field(default_factory=list)

    model_version: str
    prompt_version: str
    evaluated_at: str


class ExamAttemptEvaluationCompletedEvent(EventEnvelope):
    event_type: Literal["ExamAttemptEvaluationCompleted"] = "ExamAttemptEvaluationCompleted"

    exam_attempt_id: str
    answer_id: str
    question_id: str
    payload: ExamAttemptEvaluationCompletedPayload
