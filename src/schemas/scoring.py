from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .common import CriterionSource, ScoreStatus, _CamelMessage


class CriterionScore(_CamelMessage):
    score: Optional[float] = None
    level: str = "not_scored"
    status: ScoreStatus = "not_scored"
    source: CriterionSource = "system"
    subscores: Dict[str, Any] = Field(default_factory=dict)
    note: str = ""
    suggestion: str = ""
    weakness_labels: List[str] = Field(default_factory=list)
    evidence_spans: List[str] = Field(default_factory=list)
    recommendation_tag: str = ""
    matched_band_code: str = ""


class CriteriaScores(BaseModel):
    pronunciation: CriterionScore
    fluency: CriterionScore
    grammar: CriterionScore
    vocabulary: CriterionScore
    coherence: CriterionScore
