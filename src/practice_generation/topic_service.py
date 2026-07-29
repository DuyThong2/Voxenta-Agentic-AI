from __future__ import annotations

import os
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from config.chroma_config import settings
from vector.chroma_client import build_raw_collection


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


def propose_topics(request: TopicProposalRequest) -> TopicProposalBatch:
    client = OpenAI()
    prompt = f"""You are helping a Vietnamese high-school English speaking practice app decide
which NEW discussion topics to add for one specific learner.

Keyword evidence (counts are distinct sessions): {[
    item.model_dump() for item in request.keyword_evidence
]}
Interest scores: {request.interest_scores}
Topics already in the pool: {request.existing_topics}
Rejected topics: {request.rejected_topics}
Exhausted topics: {request.exhausted_topics}
This is a direct search keyword: {request.search_keyword}

Propose at most {request.max_proposals} new topics. Rules:
1. A topic must sustain a 10-15 minute spoken discussion for a B1-B2 learner.
2. Group related keywords into ONE topic. Do not echo a keyword as the topic.
3. At most ONE topic may go beyond observed keywords; set grounded_in_keyword=false.
4. Set evidence_type to KEYWORD, INTEREST, EXHAUSTED, or SEARCH. For KEYWORD and
   SEARCH, list only supporting input keywords in evidence_keywords.
5. Confidence must follow evidence: one session <=0.5, two <=0.7, three or more
   <=0.85; INTEREST <=0.6; EXHAUSTED <=0.7; ungrounded <=0.4; SEARCH <=0.95.
6. Every topic must differ meaningfully from pool and rejected topics. Explain in
   distinct_from.
7. Keep topics inclusive and answerable without money, travel, or specialist knowledge.
8. For direct search, return one proposal only, evidence_type=SEARCH, and confidence
   must be >=0.9 or return none.

Return the best proposals, not diverse samples. Structured data only."""
    response = client.responses.parse(
        model=os.getenv("PRACTICE_GENERATION_MODEL", "gpt-5.4"),
        reasoning={"effort": "low"},
        input=[
            {
                "role": "system",
                "content": "You design safe, explainable speaking-practice topics.",
            },
            {"role": "user", "content": prompt},
        ],
        text_format=TopicProposalBatch,
    )
    if response.output_parsed is None:
        return TopicProposalBatch(proposals=[])
    filtered = enforce_evidence_caps(response.output_parsed.proposals, request)
    filtered = _deduplicate_evidence_clusters(filtered)
    return TopicProposalBatch(
        proposals=_remove_vector_duplicates(
            filtered[: request.max_proposals],
            request,
            client,
        )
    )


def enforce_evidence_caps(
    proposals: list[TopicProposal],
    request: TopicProposalRequest,
) -> list[TopicProposal]:
    evidence_counts = {
        item.keyword.casefold(): item.session_count
        for item in request.keyword_evidence
    }
    ungrounded = 0
    filtered: list[TopicProposal] = []
    for proposal in proposals:
        if not proposal.grounded_in_keyword:
            ungrounded += 1
            proposal.confidence = min(proposal.confidence, 0.4)
            if ungrounded > 1:
                continue
        elif proposal.evidence_type in {"KEYWORD", "SEARCH"}:
            supporting_counts = [
                evidence_counts[keyword.casefold()]
                for keyword in proposal.evidence_keywords
                if keyword.casefold() in evidence_counts
            ]
            if not supporting_counts:
                continue
            max_sessions = max(supporting_counts)
            cap = 0.5 if max_sessions <= 1 else 0.7 if max_sessions == 2 else 0.85
            if proposal.evidence_type == "SEARCH":
                if not request.search_keyword or proposal.confidence < 0.9:
                    continue
                cap = 0.95
            proposal.confidence = min(proposal.confidence, cap)
        elif proposal.evidence_type == "INTEREST":
            proposal.confidence = min(proposal.confidence, 0.6)
        elif proposal.evidence_type == "EXHAUSTED":
            proposal.confidence = min(proposal.confidence, 0.7)
        filtered.append(proposal)
    return filtered


def _deduplicate_evidence_clusters(
    proposals: list[TopicProposal],
) -> list[TopicProposal]:
    result: list[TopicProposal] = []
    used_keywords: set[str] = set()
    for proposal in proposals:
        keywords = {
            keyword.casefold()
            for keyword in proposal.evidence_keywords
        }
        if (
            proposal.evidence_type == "KEYWORD"
            and proposal.grounded_in_keyword
            and keywords & used_keywords
        ):
            continue
        result.append(proposal)
        if proposal.evidence_type == "KEYWORD" and proposal.grounded_in_keyword:
            used_keywords.update(keywords)
    return result


def index_topic(request: TopicIndexRequest) -> None:
    client = OpenAI()
    text = f"{request.name}\n{request.description}".strip()
    embedding = _embed(client, text)
    metadata: dict[str, str | bool] = {
        "active": request.active,
        "status": request.status,
    }
    if request.student_id is not None:
        metadata["student_id"] = request.student_id
    collection = build_raw_collection(
        "practice_topics",
        embedding_model=settings.OPENAI_EMBEDDING_MODEL,
    )
    collection.upsert(
        ids=[request.topic_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata],
    )


def _remove_vector_duplicates(
    proposals: list[TopicProposal],
    request: TopicProposalRequest,
    client: OpenAI,
) -> list[TopicProposal]:
    if not proposals:
        return []
    collection = build_raw_collection(
        "practice_topics",
        embedding_model=settings.OPENAI_EMBEDDING_MODEL,
    )
    result: list[TopicProposal] = []
    accepted_embeddings: list[list[float]] = []
    for proposal in proposals:
        embedding = _embed(client, proposal.name)
        if _max_similarity(collection, embedding, where={"active": True}) >= 0.90:
            continue
        rejected_where = {
            "$and": [
                {"active": False},
                {"student_id": request.student_id},
            ]
        }
        if _max_similarity(collection, embedding, where=rejected_where) >= 0.90:
            continue
        if any(_cosine(embedding, previous) >= 0.90 for previous in accepted_embeddings):
            continue
        result.append(proposal)
        accepted_embeddings.append(embedding)
    return result


def _max_similarity(collection, embedding: list[float], *, where: dict) -> float:
    if collection.count() == 0:
        return 0.0
    response = collection.query(
        query_embeddings=[embedding],
        n_results=1,
        where=where,
        include=["distances"],
    )
    distances = response.get("distances") or []
    if not distances or not distances[0]:
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(distances[0][0])))


def _embed(client: OpenAI, text: str) -> list[float]:
    response = client.embeddings.create(
        model=settings.OPENAI_EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
