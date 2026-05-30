from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from .scoring import CriteriaScores
from .validity import ValidityResult


class Scores(BaseModel):
    criteria: CriteriaScores


class UIResponse(BaseModel):
    status: Optional[str] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
    question: Dict[str, Any] = Field(default_factory=dict)
    transcript: Dict[str, Any] = Field(default_factory=dict)
    validity: ValidityResult
    scores: Scores
    feedback: Dict[str, Any] = Field(default_factory=dict)
    details: Dict[str, Any] = Field(default_factory=dict)
    debug: Dict[str, Any] = Field(default_factory=dict)
