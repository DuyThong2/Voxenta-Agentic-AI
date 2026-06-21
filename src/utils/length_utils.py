from typing import Optional

from schemas.enums import QuestionType


def get_expected_min_words(
    question_type: Optional[QuestionType],
    duration_seconds: Optional[int],
    min_response_seconds: Optional[int] = None,
) -> int:
    if question_type == QuestionType.SHORT_ANSWER:
        return 3

    if min_response_seconds is not None:
        return max(1, round(min_response_seconds * 2.5))

    if duration_seconds is None:
        return 10

    if question_type == QuestionType.DESCRIPTION:
        if duration_seconds <= 30:
            return 15
        if duration_seconds <= 60:
            return 35
        return 50

    if question_type == QuestionType.OPINION:
        if duration_seconds <= 30:
            return 15
        if duration_seconds <= 60:
            return 30
        return 45

    if question_type == QuestionType.LONG_ANSWER:
        if duration_seconds <= 30:
            return 20
        if duration_seconds <= 60:
            return 40
        return 60

    return 10
