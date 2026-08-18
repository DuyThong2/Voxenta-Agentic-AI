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
    # None khi lượt nói không có bản ghi âm -- KHÔNG phải lỗi.
    #
    # Bên thi giữ lại lượt nói dù audio không tới được S3, nên
    # exam_item_response_turns.audio_url có thể rỗng. Cả chuỗi phía sau đã lo cho việc đó:
    # StartNode chấm theo transcript realtime và bỏ qua phiên âm, pronunciation_eval_node trả
    # pronunciation_error thay vì làm hỏng graph, answer_length_analysis_node chỉ đo SNR khi có
    # đường dẫn, và practiceEvalGraph/StartNode còn tự gán None vào chính field này.
    #
    # Khai báo `str` bắt buộc là mắt xích duy nhất chưa theo. Đo được 2026-08-18: một lượt im
    # lặng (không audio) làm pydantic ném ngay lúc dựng SpeakingInput, TRƯỚC khi tới những nhánh
    # đã xử lý sẵn -- retry 4 lần đều hỏng y hệt vì payload không đổi, rồi bài bị đánh
    # ExamAttemptEvaluationFailed. Cùng loại "message độc chặn hàng đợi" đã xảy ra 2026-08-15.
    audio_path: Optional[str] = None
    reference_text: Optional[str] = None
    transcribed_text: Optional[str] = None
    conversation_transcript: Optional[str] = None
    corrected_transcript: Optional[str] = None
    mode: SpeakingMode = SpeakingMode.UNSCRIPTED
    language: str = "en-US"
    criteria_frameworks: List[CriterionFramework] = Field(default_factory=list)
    question: Optional[QuestionContext] = None
    topic: Optional[TopicContext] = None
    asr_confidence: Optional[float] = None
    # Live Voice-Live transcript for this turn (see archive_store.get_realtime_transcript),
    # when start_node's caller was able to fetch one -- preferred over re-transcribing
    # audio_path via the Azure Speech SDK when present (see start_node_config.py).
    realtime_transcript: Optional[str] = None
    # ASR confidence for realtime_transcript (word-count-weighted average of each utterance's
    # C_ASR-log = sqrt(G*T20); see voice_live_client._confidence_from_logprobs / session.py).
    # None if realtime_transcript is missing or its model never reported logprobs.
    realtime_transcript_confidence: Optional[float] = None
    # SNR do được từ chính file audio của lượt này, tính NGAY khi file còn tồn tại.
    #
    # answer_length_analysis_node chỉ chạy ở pha TỔNG HỢP, sau khi mọi lượt đã chấm xong -- mà
    # file tạm của từng lượt bị os.unlink ngay trong finally của pha per-turn. Pha tổng hợp copy
    # lại speaking_input của lượt đầu nên audio_path vẫn là một chuỗi hợp lệ, chỉ có điều file ở
    # đó đã biến mất; compute_snr_db nuốt OSError và trả None, nên q_snr/q_speech/audioQuality
    # null ở MỌI bài và cổng audio chưa từng chạy -- không một dòng log lỗi nào.
    #
    # Đo sẵn ở pha per-turn rồi mang theo là cách giữ đúng ngữ nghĩa: mỗi lượt có audio riêng.
    snr_db: Optional[float] = None
    # Cùng lý do với snr_db: đo ở pha per-turn rồi mang theo, vì tới pha tổng hợp thì file đã bị
    # xoá. silence_ratio là nguồn của q_speech, clipping_ratio là tín hiệu cổng audio riêng.
    silence_ratio: Optional[float] = None
    clipping_ratio: Optional[float] = None


QuestionContext.model_rebuild()
