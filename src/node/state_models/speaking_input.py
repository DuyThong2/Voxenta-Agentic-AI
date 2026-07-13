from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from schemas.evaluation_event import EvaluationGuideInput
from schemas.framework import CriterionFramework
from schemas.enums import DifficultyLevel, QuestionType, SpeakingMode


class QuestionContext(BaseModel):
    """Question context carried through the evaluation graphs.

    Question metadata is authored by school/teacher users on the Java side and
    forwarded here as-is — treat it as untrusted input. Unrecognized enum
    values are dropped to None rather than raising, and an inconsistent
    response-time window (negative, zero, or min > max) is dropped entirely
    rather than guessed at.
    """

    question_text: Optional[str] = None
    question_type: Optional[QuestionType] = None
    difficulty_level: Optional[DifficultyLevel] = None
    duration_seconds: Optional[int] = None
    min_response_seconds: Optional[int] = None
    max_response_seconds: Optional[int] = None
    evaluation_guide: Optional[EvaluationGuideInput] = None
    asset: Optional["QuestionAssetContext"] = None

    @field_validator("question_type", mode="before")
    @classmethod
    def _tolerant_question_type(cls, value: Any) -> Any:
        if value is None or isinstance(value, QuestionType):
            return value
        try:
            return QuestionType(value)
        except ValueError:
            return None

    @field_validator("difficulty_level", mode="before")
    @classmethod
    def _tolerant_difficulty_level(cls, value: Any) -> Any:
        if value is None or isinstance(value, DifficultyLevel):
            return value
        try:
            return DifficultyLevel(value)
        except ValueError:
            return None

    @model_validator(mode="after")
    def _sanitize_response_window(self) -> "QuestionContext":
        if self.min_response_seconds is not None and self.min_response_seconds <= 0:
            self.min_response_seconds = None
        if self.max_response_seconds is not None and self.max_response_seconds <= 0:
            self.max_response_seconds = None
        if (
            self.min_response_seconds is not None
            and self.max_response_seconds is not None
            and self.min_response_seconds > self.max_response_seconds
        ):
            self.min_response_seconds = None
            self.max_response_seconds = None
        return self


class QuestionAssetContext(BaseModel):
    type: Optional[str] = None
    transcript: Optional[str] = None
    description: Optional[str] = None
    alt_text: Optional[str] = None


class TopicContext(BaseModel):
    """Topic context for evaluation."""

    topic_id: Optional[int] = None
    topic_name: Optional[str] = None
    topic_description: Optional[str] = None


class SpeakingInput(BaseModel):
    exam_attempt_id: Optional[str] = None
    answer_id: Optional[str] = None
    question_id: Optional[str] = None
    audio_path: str
    reference_text: Optional[str] = None
    transcribed_text: Optional[str] = None
    conversation_transcript: Optional[str] = None
    corrected_transcript: Optional[str] = None
    mode: SpeakingMode = SpeakingMode.UNSCRIPTED
    language: str = "en-US"
    criteria_frameworks: List[CriterionFramework] = Field(default_factory=list)
    question: Optional[QuestionContext] = None
    topic: Optional[TopicContext] = None
    answer_length_metrics: Optional[dict] = None


QuestionContext.model_rebuild()
