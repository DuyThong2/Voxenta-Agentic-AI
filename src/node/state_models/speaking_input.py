from typing import List, Optional

from pydantic import BaseModel, Field

from schemas.evaluation_event import EvaluationGuideInput
from schemas.framework import CriterionFramework
from schemas.enums import DifficultyLevel, QuestionType, SpeakingMode


class QuestionContext(BaseModel):
    """Question context carried through the evaluation graphs."""

    question_text: Optional[str] = None
    question_type: Optional[QuestionType] = None
    difficulty_level: Optional[DifficultyLevel] = None
    duration_seconds: Optional[int] = None
    min_response_seconds: Optional[int] = None
    max_response_seconds: Optional[int] = None
    evaluation_guide: Optional[EvaluationGuideInput] = None


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
    corrected_transcript: Optional[str] = None
    mode: SpeakingMode = SpeakingMode.UNSCRIPTED
    language: str = "en-US"
    criteria_frameworks: List[CriterionFramework] = Field(default_factory=list)
    question: Optional[QuestionContext] = None
    topic: Optional[TopicContext] = None
    answer_length_metrics: Optional[dict] = None
