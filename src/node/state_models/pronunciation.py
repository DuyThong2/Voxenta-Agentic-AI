from typing import List, Literal, Optional, Dict, Any

from pydantic import BaseModel, Field

from schemas.scoring import CriteriaScores


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


class FormattedPronunciationResult(BaseModel):
    recognized_text: Optional[str] = None
    reference_text: Optional[str] = None
    mode: Optional[str] = None
    overall: Dict[str, Optional[float]] = Field(default_factory=dict)
    criteria: CriteriaScores = Field(default_factory=lambda: CriteriaScores(
        pronunciation={}, fluency={}, grammar={}, vocabulary={}, coherence={},
    ))
    correction_summary: Dict[str, Any] = Field(default_factory=dict)
    word_feedback: List[Dict[str, Any]] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    raw_result: Optional[Dict[str, Any]] = None