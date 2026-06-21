from typing import List, Optional

from pydantic import Field

from schemas.common import CriterionName, _CamelMessage


class FrameworkBand(_CamelMessage):
    code: str
    label: Optional[str] = None
    score_min: float
    score_max: float
    descriptor: Optional[str] = None
    positive_signals: List[str] = Field(default_factory=list)
    negative_signals: List[str] = Field(default_factory=list)


class CriterionFramework(_CamelMessage):
    criterion_key: CriterionName
    framework_code: Optional[str] = None
    framework_criterion_name: Optional[str] = None
    framework_criterion_description: Optional[str] = None
    rubric_weight: Optional[float] = None
    rubric_min_score: float = 0
    rubric_max_score: float = 100
    bands: List[FrameworkBand] = Field(default_factory=list)
