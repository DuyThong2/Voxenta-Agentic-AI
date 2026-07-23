from typing import List, Optional

from pydantic import Field

from node.state_models.pronunciation import WordFeedback
from schemas.common import _CamelMessage


class TurnInput(_CamelMessage):
    turn_order: int
    turn_type: str
    prompt_text: Optional[str] = None
    audio_ref: str
    transcript: Optional[str] = None
    duration_seconds: Optional[float] = None


class PronunciationOverallScores(_CamelMessage):
    accuracy_score: Optional[float] = None
    fluency_score: Optional[float] = None
    prosody_score: Optional[float] = None
    pron_score: Optional[float] = None
    completeness_score: Optional[float] = None


class TurnDetail(_CamelMessage):
    turn_id: Optional[str] = None
    turn_order: int
    turn_type: str
    prompt_text: Optional[str] = None
    audio_url: str
    transcript: str
    word_count: int
    duration_seconds: Optional[float] = None
    asr_confidence: Optional[float] = None
    pronunciation_overall: PronunciationOverallScores = Field(default_factory=PronunciationOverallScores)
    word_feedback: List[WordFeedback] = Field(default_factory=list)


class ConfidenceCaseSignals(_CamelMessage):
    c_asr_log: Optional[float] = None
    cross_asr_agreement: Optional[float] = None
    q_snr: Optional[float] = None
    q_speech: Optional[float] = None
    clipping_ratio: Optional[float] = None
    c_ref: Optional[float] = None
    c_align: Optional[float] = None
    c_pf_branch: Optional[float] = None
    c_grammar: Optional[float] = None
    c_vocabulary: Optional[float] = None
    c_discourse: Optional[float] = None


class EvaluationSignals(_CamelMessage):
    duration_seconds: int = 0
    word_count: int
    sentence_count: int
    length_ratio: Optional[float] = None
    expected_min_words: int
    asr_confidence_avg: Optional[float] = None
    topic_relevance_score: Optional[float] = None
    off_topic_ratio: Optional[float] = None
    code_switching_ratio: Optional[float] = None
    speech_rate: Optional[float] = None
    ai_confidence: Optional[float] = None
    audio_quality: Optional[float] = None
    silence_ratio: Optional[float] = None
    confidence_case: Optional[ConfidenceCaseSignals] = None
