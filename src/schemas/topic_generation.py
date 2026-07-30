from typing import Literal

from pydantic import BaseModel, Field


class KeywordEvidence(BaseModel):
    keyword: str
    session_count: int = Field(ge=0)


class TopicProposalRequest(BaseModel):
    student_id: str
    keyword_evidence: list[KeywordEvidence]
    interest_scores: dict[str, float] = Field(default_factory=dict)
    existing_topics: list[str] = Field(default_factory=list)
    rejected_topics: list[str] = Field(default_factory=list)
    exhausted_topics: list[str] = Field(default_factory=list)
    search_keyword: bool = False
    max_proposals: int = Field(default=3, ge=1, le=3)


class TopicProposal(BaseModel):
    name: str
    interest_dimension: Literal[
        "ENTERTAINMENT_MEDIA",
        "TECH_GAMING",
        "SPORTS_HEALTH",
        "PEOPLE_SOCIETY",
        "TRAVEL_PLACES",
        "FUTURE_SCIENCE",
    ]
    curriculum_group: Literal["IN_GDPT2018", "OUT_OF_CURRICULUM"]
    confidence: float = Field(ge=0, le=1)
    reason_text: str = Field(min_length=1)
    distinct_from: str = Field(min_length=1)
    evidence_type: Literal["KEYWORD", "INTEREST", "EXHAUSTED", "SEARCH"]
    evidence_keywords: list[str] = Field(default_factory=list)
    grounded_in_keyword: bool


class TopicProposalBatch(BaseModel):
    proposals: list[TopicProposal] = Field(max_length=3)


class TopicIndexRequest(BaseModel):
    topic_id: str
    name: str
    description: str = ""
    active: bool
    student_id: str | None = None
    status: Literal["ACTIVE", "REJECTED"] = "ACTIVE"
