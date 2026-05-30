from typing import Literal

AssessmentAction = Literal["score", "score_with_penalty", "teacher_review", "reject_or_zero"]
Severity = Literal["none", "low", "medium", "high", "critical"]
CriterionName = Literal["pronunciation", "fluency", "grammar", "vocabulary", "coherence"]
CriterionSource = Literal["azure", "llm", "rule", "system"]
ScoreStatus = Literal["scored", "not_scored", "zeroed", "capped"]
