from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class _CamelMessage(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

AssessmentAction = Literal["score", "score_with_penalty", "teacher_review", "reject_or_zero"]
Severity = Literal["none", "low", "medium", "high", "critical"]
CriterionName = Literal["pronunciation", "fluency", "grammar", "vocabulary", "coherence"]
CriterionSource = Literal["azure", "llm", "rule", "system"]
ScoreStatus = Literal["scored", "not_scored", "zeroed", "capped"]
