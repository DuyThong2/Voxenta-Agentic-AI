from typing import Any, Dict, List, Optional
from typing import Literal

from pydantic import BaseModel, Field

from .common import AssessmentAction, CriterionName, Severity


class RuleResult(BaseModel):
    rule_id: str
    category: str
    status: Literal["triggered", "not_triggered", "not_evaluated"] = "triggered"
    severity: Severity = "none"
    action: AssessmentAction = "score"
    blocking: bool = False
    target_criteria: List[CriterionName] = Field(default_factory=list)
    message: str = ""
    evidence: Dict[str, Any] = Field(default_factory=dict)
    score_caps: Dict[str, Optional[float]] = Field(default_factory=dict)
    penalties: List[Dict[str, Any]] = Field(default_factory=list)
    suggestion: str = ""


class ValidityResult(BaseModel):
    valid_for_scoring: bool = True
    action: AssessmentAction = "score"
    overall_severity: Severity = "none"
    rule_results: List[RuleResult] = Field(default_factory=list)
    flags: List[Dict[str, Any]] = Field(default_factory=list)
    score_caps: Dict[str, Optional[float]] = Field(default_factory=dict)
    penalties: List[Dict[str, Any]] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    transcript_source: str = ""
    transcript_word_count: int = 0
