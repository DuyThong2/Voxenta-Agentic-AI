from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, create_model

from schemas.interest_quiz_generation import DEFAULT_INTEREST_DIMENSIONS


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
    # Danh mục chiều hiện hành do Java gửi xuống (bảng interest_dimension); rỗng thì dùng mặc định.
    dimensions: list[str] = Field(default_factory=list)

    def effective_dimensions(self) -> list[str]:
        return self.dimensions or DEFAULT_INTEREST_DIMENSIONS


class TopicProposal(BaseModel):
    """Kiểu 'mở' cho đường đọc. Đường SINH dùng build_topic_batch_model để ràng buộc
    interest_dimension theo đúng danh mục hiện hành ngay lúc decode."""

    name: str
    interest_dimension: str
    curriculum_group: Literal["IN_GDPT2018", "OUT_OF_CURRICULUM"]
    confidence: float = Field(ge=0, le=1)
    reason_text: str = Field(min_length=1)
    distinct_from: str = Field(min_length=1)
    evidence_type: Literal["KEYWORD", "INTEREST", "EXHAUSTED", "SEARCH"]
    evidence_keywords: list[str] = Field(default_factory=list)
    grounded_in_keyword: bool


class TopicProposalBatch(BaseModel):
    proposals: list[TopicProposal] = Field(max_length=3)


def build_topic_batch_model(dimensions: list[str]) -> type[BaseModel]:
    """Xem build_quiz_batch_model trong schemas/interest_quiz_generation.py để biết vì sao
    dựng động thay vì để str + lọc hậu kiểm."""
    dimension_enum = Enum(  # type: ignore[misc]
        "DynamicTopicDimension",
        {code: code for code in dimensions},
        type=str,
    )
    proposal_model = create_model(
        "DynamicTopicProposal",
        name=(str, ...),
        interest_dimension=(dimension_enum, ...),  # type: ignore[valid-type]
        curriculum_group=(Literal["IN_GDPT2018", "OUT_OF_CURRICULUM"], ...),
        confidence=(float, Field(ge=0, le=1)),
        reason_text=(str, Field(min_length=1)),
        distinct_from=(str, Field(min_length=1)),
        evidence_type=(
            Literal["KEYWORD", "INTEREST", "EXHAUSTED", "SEARCH"],
            ...,
        ),
        evidence_keywords=(list[str], Field(default_factory=list)),
        grounded_in_keyword=(bool, ...),
    )
    return create_model(
        "DynamicTopicProposalBatch",
        proposals=(list[proposal_model], Field(max_length=3)),  # type: ignore[valid-type]
    )


class TopicIndexRequest(BaseModel):
    topic_id: str
    name: str
    description: str = ""
    active: bool
    student_id: str | None = None
    status: Literal["ACTIVE", "REJECTED"] = "ACTIVE"
