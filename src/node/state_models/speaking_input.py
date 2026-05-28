from typing import Literal, Optional

from pydantic import BaseModel


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

    mode: Literal["scripted", "unscripted"] = "unscripted"
    language: str = "en-US"

    # Question context — được truyền từ .NET backend khi evaluate.
    # LLM evaluator dùng để chấm điểm chính xác hơn.
    question_id: Optional[int] = None
    question_text: Optional[str] = None
    question_type: Optional[str] = None  # read_aloud, short_answer, long_answer, opinion, description
    difficulty_level: Optional[str] = None  # easy, medium, hard
    duration_seconds: Optional[int] = None

    # Answer length analysis for development scoring.
    answer_length_metrics: Optional[dict] = None

    # Topic context
    topic_id: Optional[int] = None
    topic_name: Optional[str] = None
    topic_description: Optional[str] = None