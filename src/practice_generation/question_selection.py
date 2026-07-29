from __future__ import annotations

import math

from pydantic import BaseModel, Field

from config.chroma_config import settings
from vector.chroma_client import build_raw_collection


class SimilarityRequest(BaseModel):
    candidate_ids: list[str] = Field(max_length=50)
    selected_ids: list[str] = Field(max_length=20)


class SimilarityResponse(BaseModel):
    max_similarity: dict[str, float]


def max_similarities(request: SimilarityRequest) -> SimilarityResponse:
    if not request.candidate_ids or not request.selected_ids:
        return SimilarityResponse(
            max_similarity={
                candidate_id: 0.0
                for candidate_id in request.candidate_ids
            }
        )
    collection = build_raw_collection(
        "practice_questions",
        embedding_model=settings.OPENAI_EMBEDDING_MODEL,
    )
    ids = list(dict.fromkeys(
        request.candidate_ids + request.selected_ids
    ))
    records = collection.get(ids=ids, include=["embeddings"])
    record_ids = records.get("ids")
    record_embeddings = records.get("embeddings")
    embeddings = dict(zip(
        [] if record_ids is None else record_ids,
        [] if record_embeddings is None else record_embeddings,
        strict=True,
    ))
    selected = [
        embeddings[item]
        for item in request.selected_ids
        if item in embeddings
    ]
    result: dict[str, float] = {}
    for candidate_id in request.candidate_ids:
        candidate = embeddings.get(candidate_id)
        if candidate is None or not selected:
            result[candidate_id] = 1.0
            continue
        result[candidate_id] = max(
            _cosine(candidate, selected_embedding)
            for selected_embedding in selected
        )
    return SimilarityResponse(max_similarity=result)


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
