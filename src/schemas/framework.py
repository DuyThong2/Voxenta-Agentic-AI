import math
from typing import List, Optional

from pydantic import Field, model_validator

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
    # (see each eval node's *_node_helper.build_framework_criterion_context).
    criterion_key: str
    framework_code: Optional[str] = None
    framework_criterion_name: Optional[str] = None
    framework_criterion_description: Optional[str] = None
    target_band_id: Optional[str] = None
    target_band_code: Optional[str] = None
    target_band_label: Optional[str] = None
    target_band_only: bool = False
    rubric_weight: Optional[float] = None
    rubric_min_score: float = 0
    rubric_max_score: float = 100
    bands: List[FrameworkBand] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_target_band_scope(self):
        if (
            not math.isfinite(self.rubric_min_score)
            or not math.isfinite(self.rubric_max_score)
            or self.rubric_max_score <= self.rubric_min_score
        ):
            raise ValueError(
                "rubric_max_score must be greater than rubric_min_score"
            )
        if not self.target_band_only:
            return self
        if not self.target_band_code:
            raise ValueError("target_band_code is required when target_band_only is true")
        if len(self.bands) != 1:
            raise ValueError("target-band-only scoring must receive exactly one framework band")

        target_band = self.bands[0]
        if target_band.code != self.target_band_code:
            raise ValueError("the only framework band must match target_band_code")
        if (
            target_band.score_min != self.rubric_min_score
            or target_band.score_max != self.rubric_max_score
        ):
            raise ValueError(
                "the target band must cover the complete rubric score range"
            )
        return self
