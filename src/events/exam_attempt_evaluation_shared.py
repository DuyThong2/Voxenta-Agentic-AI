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
    q_snr: Optional[float] = None
    q_speech: Optional[float] = None
    clipping_ratio: Optional[float] = None
    c_ref: Optional[float] = None
    # TẮT 2026-08-11 -- case (4), xem khối chú thích ở utils/confidence_utils.py.
    #
    # Bỏ hẳn khỏi payload chứ không gửi null: 4 trường này của Java (ConfidenceCaseSignalsDto)
    # là Double nên thiếu khoá cũng ra null, đúng thứ nhóm D cần để tự bỏ qua. Giữ lại mà luôn
    # gửi null thì mỗi bản ghi signals lưu trong DB lại thêm 4 khoá rỗng vô nghĩa.
    #
    # c_align: Optional[float] = None
    # 3 thành phần RIÊNG của case (4) -- xem AlignmentConfidence trong confidence_utils.py.
    # Ngưỡng Vietnam-adjusted (m soft>0.20/hard>0.30, c soft<0.90/hard<0.80, j soft>0.15/hard>0.30)
    # áp RIÊNG cho từng cái, KHÔNG áp chung 1 ngưỡng lên c_align (composite) -- c_align ở trên chỉ
    # còn dùng để hiển thị/tính cPfBranch, không dùng để quyết định review nữa.
    # c_align_accuracy: Optional[float] = None
    # c_align_coverage: Optional[float] = None
    # c_align_timing: Optional[float] = None
    c_pf_branch: Optional[float] = None
    c_grammar: Optional[float] = None
    c_vocabulary: Optional[float] = None
    c_coherence: Optional[float] = None
    grammar_score_delta: Optional[float] = None
    vocabulary_score_delta: Optional[float] = None
    coherence_score_delta: Optional[float] = None


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
    audio_quality: Optional[float] = None
    silence_ratio: Optional[float] = None
    evidence_status: str = "SUFFICIENT"
    evidence_reason_codes: List[str] = Field(default_factory=list)
    confidence_case: Optional[ConfidenceCaseSignals] = None
