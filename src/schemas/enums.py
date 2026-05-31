"""
Centralized enums for pronunciation evaluation system.

These replace raw string literals throughout the codebase to prevent typos
and enable IDE autocomplete / static checking.
"""

from enum import Enum


class QuestionType(str, Enum):
    """Question type for speaking assessment."""
    READ_ALOUD = "read_aloud"
    SHORT_ANSWER = "short_answer"
    LONG_ANSWER = "long_answer"
    OPINION = "opinion"
    DESCRIPTION = "description"


class DifficultyLevel(str, Enum):
    """Question difficulty level."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class LengthCategory(str, Enum):
    """Answer length category from answer-length analysis node."""
    TOO_SHORT = "too_short"
    SOMEWHAT_SHORT = "somewhat_short"
    APPROPRIATE = "appropriate"
    TOO_LONG_OR_VERBOSE = "too_long_or_verbose"


class SpeakingMode(str, Enum):
    """Speaking evaluation mode."""
    SCRIPTED = "scripted"
    UNSCRIPTED = "unscripted"


class ScoreColor(str, Enum):
    """Color band for pronunciation scores."""
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    GRAY = "gray"

    @classmethod
    def from_score(cls, score: float | None) -> "ScoreColor":
        """Map a numeric score (0-100) to a color band."""
        if score is None:
            return cls.GRAY
        if score < 60:
            return cls.RED
        if score < 80:
            return cls.YELLOW
        return cls.GREEN
