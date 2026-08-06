from typing import Any, TypedDict

from schemas.question_generation import (
    CandidateVerdict,
    PracticeQuestionCandidate,
)


class QuestionGenerationState(TypedDict, total=False):
    topic: tuple[str, str, str]
    criterion: tuple[str, str | None]
    target_rank: int
    # Đường online (học sinh đang chờ): cắt bớt các bước chỉ phục vụ đo đạc /
    # đánh bóng. Xem constants.FAST_* để biết chính xác bỏ gì và vì sao.
    fast: bool
    # So bac + mo ta thang cua framework dang ap (Java gui xuong). Xem constants.BAND_LADDER
    # de biet vi sao khong dung hang so 6 bac cua VSTEP nua.
    band_count: int
    band_ladder: list[Any]
    # Id cau da CHET VINH VIEN voi hoc sinh dang cho -- CandidateFilterNode bo chung ra khoi
    # phep so trung. Xem runtime.max_similarity de biet vi sao khong so voi ca kho.
    exclude_question_ids: set[str]
    # Số câu thực sự cần -- fast mode dừng sửa/chấm ngay khi đã đủ.
    needed: int
    candidates: list[PracticeQuestionCandidate]
    survivors: list[PracticeQuestionCandidate]
    survivor_embeddings: dict[str, list[float]]
    separate_verdicts: dict[str, CandidateVerdict]
    live: list[PracticeQuestionCandidate]
    refined: list[PracticeQuestionCandidate]
    rejected: list[dict[str, Any]]
    filter_reasons: set[str]
    drafter_raw: dict[str, Any]
    evaluator_raw: dict[str, Any]
    editor_raw: list[dict[str, Any]]
    token_calls: list[Any]
    cosines: list[float]
    evaluator_rejected: int
    comparison_total: int
    comparison_different: int
    editor_rounds: list[int]
