from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, create_model

from schemas.interest_quiz_generation import DEFAULT_INTEREST_DIMENSIONS


# Trần số đề xuất một lượt. Java xin NHIỀU hơn số chủ đề thực cần vì bộ lọc trùng-gần của nó
# cắt rất mạnh (xem TopicSuggestionService.synchronousOffers). Con số này phải khớp ở CẢ BA
# nơi bên dưới -- request, response, và schema decode gửi cho LLM; để lệch thì hoặc 422 ở cửa,
# hoặc LLM bị ép trả ít hơn số đã xin mà không ai thấy.
MAX_TOPIC_PROPOSALS = 8


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
    max_proposals: int = Field(default=3, ge=1, le=MAX_TOPIC_PROPOSALS)
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
    # Chu de nay tu nhien goi ra khung thoi gian nao. Java doc de quyet thi dich cho tung cau
    # (TensePolicy.forSlot): PAST/FUTURE thi khoa, MIXED thi xoay vong qua cac thi theo o.
    #
    # Hoi o day chu khong doan ve sau: ngay luc soan, mo hinh dang hieu chu de noi ve cai gi.
    # Doan lai tu ten chu de thi khong co co so, va doan sai thi moi cau cua chu de do bi ep
    # sai khung thoi gian ma khong ai thay.
    temporal_affordance: Literal["PAST", "FUTURE", "MIXED"] = "MIXED"
    confidence: float = Field(ge=0, le=1)
    reason_text: str = Field(min_length=1)
    distinct_from: str = Field(min_length=1)
    evidence_type: Literal["KEYWORD", "INTEREST", "EXHAUSTED", "SEARCH"]
    evidence_keywords: list[str] = Field(default_factory=list)
    grounded_in_keyword: bool


class TopicProposalBatch(BaseModel):
    proposals: list[TopicProposal] = Field(max_length=MAX_TOPIC_PROPOSALS)


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
        temporal_affordance=(Literal["PAST", "FUTURE", "MIXED"], ...),
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
        proposals=(list[proposal_model], Field(max_length=MAX_TOPIC_PROPOSALS)),  # type: ignore[valid-type]
    )


class TopicIndexRequest(BaseModel):
    topic_id: str
    name: str
    description: str = ""
    active: bool
    student_id: str | None = None
    status: Literal["ACTIVE", "REJECTED"] = "ACTIVE"


class TopicSearchRequest(BaseModel):
    """Tim chu de theo NGU NGHIA -- bo sung cho tim theo chuoi ben Java, khong thay the no."""

    keyword: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=10, ge=1, le=50)
    # Duoi nguong nay thi coi nhu khong lien quan. 0.55 la diem bat dau (do tren
    # text-embedding-3-large); can chinh theo thuc te chu khong phai hang so thieng.
    min_similarity: float = Field(default=0.55, ge=0.0, le=1.0)


class TopicSearchHit(BaseModel):
    topic_id: str
    similarity: float


class TopicSearchResponse(BaseModel):
    """Chi tra ID. Ten/mo ta trong vector store co the da cu -- Java tu doc lai tu Postgres."""

    hits: list[TopicSearchHit]
