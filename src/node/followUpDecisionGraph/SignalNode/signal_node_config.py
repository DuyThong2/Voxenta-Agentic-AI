from typing import Any, Dict, List

from node.followUpDecisionGraph.constants import MAX_TURNS
from node.state_models import QuestionContext
from utils.length_utils import get_expected_min_words


def _question_attr(question: QuestionContext | Dict[str, Any] | None, key: str) -> Any:
    if question is None:
        return None
    if isinstance(question, dict):
        return question.get(key)
    return getattr(question, key, None)


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

    turn_order = state.get("turn_order") or current_turn.get("turn_order") or 0
    hard_stop_reason = None
    if turn_order >= MAX_TURNS:
        hard_stop_reason = "max_turns_reached"
    elif cumulative_word_count == 0:
        hard_stop_reason = "no_speech"

    return {
        **state,
        "status": "processing",
        "turns": [current_turn],
        "signals": {
            "expected_min_words": expected_min_words,
            "cumulative_word_count": cumulative_word_count,
            "length_sufficient": length_sufficient,
            "hard_stop": hard_stop_reason is not None,
            "hard_stop_reason": hard_stop_reason,
        },
    }
