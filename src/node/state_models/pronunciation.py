from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class PhonemeFeedback(BaseModel):
    phoneme: str
    accuracy_score: Optional[float] = None
    color: Optional[Literal["red", "yellow", "green", "gray"]] = None


class WordFeedback(BaseModel):
    word: str
    accuracy_score: Optional[float] = None
    error_type: Optional[str] = None
    color: Optional[Literal["red", "yellow", "green", "gray"]] = None
    phonemes: List[PhonemeFeedback] = Field(default_factory=list)


class PronunciationAssessmentResult(BaseModel):
    recognized_text: Optional[str] = None

    accuracy_score: Optional[float] = None
    fluency_score: Optional[float] = None
    prosody_score: Optional[float] = None
    pron_score: Optional[float] = None
    completeness_score: Optional[float] = None

    word_feedback: List[WordFeedback] = Field(default_factory=list)

    # Để debug, lúc production có thể bỏ/lưu DB riêng
    raw_result: Optional[dict] = None