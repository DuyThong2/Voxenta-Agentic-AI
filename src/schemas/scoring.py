from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from .common import CriterionSource, ScoreStatus


class CriterionScore(BaseModel):
    score: Optional[float] = None
    level: str = "not_scored"
    status: ScoreStatus = "not_scored"
    source: CriterionSource = "system"
    subscores: Dict[str, Any] = Field(default_factory=dict)
    note: str = ""
    suggestion: str = ""


class CriteriaScores(BaseModel):
    pronunciation: CriterionScore
    fluency: CriterionScore
    grammar: CriterionScore
    vocabulary: CriterionScore
    coherence: CriterionScore
