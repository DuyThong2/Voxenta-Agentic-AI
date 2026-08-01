from typing import Any, TypedDict

from schemas.question_generation import (
    CandidateVerdict,
    PracticeQuestionCandidate,
)


class QuestionGenerationState(TypedDict, total=False):
    topic: tuple[str, str, str]
    criterion: tuple[str, str | None]
    target_rank: int
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
