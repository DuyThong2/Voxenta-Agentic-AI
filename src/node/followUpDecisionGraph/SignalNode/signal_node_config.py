from typing import Any, Dict, List

from node.followUpDecisionGraph.constants import MAX_TURNS
from node.followUpDecisionGraph.followup_graph_helper import count_assessment_turns
from node.state_models import QuestionContext
from utils.length_utils import get_expected_min_words

_WORDS_PER_SECOND = 2.5


def _question_attr(question: QuestionContext | Dict[str, Any] | None, key: str) -> Any:
    if question is None:
        return None
    if isinstance(question, dict):
        return question.get(key)
    return getattr(question, key, None)


def _resolve_target_response_seconds(question: QuestionContext | Dict[str, Any] | None) -> int | None:
    min_response_seconds = _question_attr(question, "min_response_seconds")
    max_response_seconds = _question_attr(question, "max_response_seconds")
    duration_seconds = _question_attr(question, "duration_seconds")

    if min_response_seconds is not None and min_response_seconds > 0:
        return int(min_response_seconds)
    if duration_seconds is not None and duration_seconds > 0:
        return int(duration_seconds)
    if max_response_seconds is not None and max_response_seconds > 0:
        return int(max_response_seconds)
    return None


def _estimate_speaking_seconds(word_count: int) -> float:
    return round(word_count / _WORDS_PER_SECOND, 1)


def _followup_pressure(
    *,
    length_sufficient: bool,
    coverage_ratio: float | None,
    assessment_turn_count: int,
) -> str:
    if length_sufficient and (
        assessment_turn_count >= 2 or (coverage_ratio is not None and coverage_ratio >= 1.15)
    ):
        return "high"
    if assessment_turn_count >= 2 or (coverage_ratio is not None and coverage_ratio >= 0.9):
        return "medium"
    return "low"


def prepare_turn_signals_node(state: Dict[str, Any]) -> Dict[str, Any]:
    current_turn = state.get("current_turn")
    if current_turn is None:
        return {
            **state,
            "status": "error",
            "error": "current_turn is required for prepare_turn_signals_node",
        }

    question = state.get("question")
    previous_turns: List[Dict[str, Any]] = list(state.get("turns", []))
    all_turns = [*previous_turns, current_turn]

    expected_min_words = get_expected_min_words(
        _question_attr(question, "question_type"),
        _question_attr(question, "duration_seconds"),
        min_response_seconds=_question_attr(question, "min_response_seconds"),
    )
    cumulative_word_count = sum(int((turn or {}).get("word_count") or 0) for turn in all_turns)
    length_sufficient = cumulative_word_count >= expected_min_words
    current_turn_word_count = int((current_turn or {}).get("word_count") or 0)
    estimated_response_seconds = _estimate_speaking_seconds(cumulative_word_count)
    actual_response_seconds = sum(float((turn or {}).get("duration_seconds") or 0) for turn in all_turns) or None
    target_response_seconds = _resolve_target_response_seconds(question)
    word_coverage_ratio = (
        round(estimated_response_seconds / target_response_seconds, 2)
        if target_response_seconds
        else None
    )
    time_coverage_ratio = (
        round(actual_response_seconds / target_response_seconds, 2)
        if target_response_seconds and actual_response_seconds is not None
        else None
    )
    response_coverage_ratio = min(
        (ratio for ratio in (word_coverage_ratio, time_coverage_ratio) if ratio is not None),
        default=None,
    )

    assessment_turn_count = count_assessment_turns(previous_turns) + 1
    hard_stop_reason = None
    if assessment_turn_count >= MAX_TURNS:
        hard_stop_reason = "max_turns_reached"

    return {
        **state,
        "status": "processing",
        "turns": [current_turn],
        "signals": {
            "expected_min_words": expected_min_words,
            "cumulative_word_count": cumulative_word_count,
            "current_turn_word_count": current_turn_word_count,
            "length_sufficient": length_sufficient,
            "assessment_turn_count": assessment_turn_count,
            "estimated_response_seconds": estimated_response_seconds,
            "actual_response_seconds": actual_response_seconds,
            "target_response_seconds": target_response_seconds,
            "word_coverage_ratio": word_coverage_ratio,
            "time_coverage_ratio": time_coverage_ratio,
            "response_coverage_ratio": response_coverage_ratio,
            "followup_pressure": _followup_pressure(
                length_sufficient=length_sufficient,
                coverage_ratio=response_coverage_ratio,
                assessment_turn_count=assessment_turn_count,
            ),
            "no_meaningful_speech": current_turn_word_count == 0,
            "hard_stop": hard_stop_reason is not None,
            "hard_stop_reason": hard_stop_reason,
        },
    }
