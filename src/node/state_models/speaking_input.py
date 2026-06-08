from typing import Optional

from pydantic import BaseModel

from schemas.enums import DifficultyLevel, QuestionType, SpeakingMode


class QuestionContext(BaseModel):
    """Question context — được truyền từ .NET backend khi evaluate."""
    question_id: Optional[int] = None
    question_text: Optional[str] = None
    question_type: Optional[QuestionType] = None
    difficulty_level: Optional[DifficultyLevel] = None
    duration_seconds: Optional[int] = None


class TopicContext(BaseModel):
    """Topic context for evaluation."""
    topic_id: Optional[int] = None
    topic_name: Optional[str] = None
    topic_description: Optional[str] = None


class SpeakingInput(BaseModel):
    audio_path: str

    # Có reference_text thì scripted.
    # Không có reference_text thì unscripted.
    # Trong scripted mode, reference_text là nguồn dữ liệu chính xác nhất.
    reference_text: Optional[str] = None

    # Raw transcript from the speech recognizer.
    # Start node ghi vào đây trước khi correction node xử lý.
    transcribed_text: Optional[str] = None

    # Corrected transcript after correction node.
    # Chỉ dùng khi không có reference_text (unscripted mode).
    corrected_transcript: Optional[str] = None

    mode: SpeakingMode = SpeakingMode.UNSCRIPTED
    language: str = "en-US"

    # Nested question/topic context
    question: Optional[QuestionContext] = None
    topic: Optional[TopicContext] = None

    # Answer length analysis for development scoring.
    answer_length_metrics: Optional[dict] = None
