from typing import List, Optional

from pydantic import Field

from schemas.common import _CamelMessage


class FrameworkBand(_CamelMessage):
    code: str
    label: Optional[str] = None
    score_min: float
    score_max: float
    descriptor: Optional[str] = None
    positive_signals: List[str] = Field(default_factory=list)
    negative_signals: List[str] = Field(default_factory=list)


class CriterionFramework(_CamelMessage):
    # Plain str, not the CriterionName Literal: this is authored on the Java
    # side and forwarded as-is. An unrecognized key must not fail parsing of
    # the whole event — it simply won't match any node's lookup downstream
    # (see utils.framework_context_helper.build_framework_criterion_context).
    criterion_key: str
    framework_code: Optional[str] = None
    framework_criterion_name: Optional[str] = None
    framework_criterion_description: Optional[str] = None
    rubric_weight: Optional[float] = None
    rubric_min_score: float = 0
    rubric_max_score: float = 100
    bands: List[FrameworkBand] = Field(default_factory=list)
