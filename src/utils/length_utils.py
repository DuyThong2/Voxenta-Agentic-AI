from typing import Optional


def get_expected_min_words(question_type: Optional[str], duration_seconds: Optional[int]) -> int:
    if question_type == "short_answer":
        return 3

    if duration_seconds is None:
        return 10

    if question_type == "description":
        if duration_seconds <= 30:
            return 15
        if duration_seconds <= 60:
            return 35
        return 50

    if question_type == "opinion":
        if duration_seconds <= 30:
            return 15
        if duration_seconds <= 60:
            return 30
        return 45

    if question_type == "long_answer":
        if duration_seconds <= 30:
            return 20
        if duration_seconds <= 60:
            return 40
        return 60

    return 10
