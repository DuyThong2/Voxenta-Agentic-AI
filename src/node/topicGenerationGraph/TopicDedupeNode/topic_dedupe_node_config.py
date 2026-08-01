from openai import OpenAI

from config.chroma_config import settings
from node.topicGenerationGraph.constants import DUPLICATE_THRESHOLD
from node.topicGenerationGraph.GraphState import TopicGenerationState
from vector.chroma_client import build_raw_collection


def topic_dedupe_node(
    state: TopicGenerationState,
    client: OpenAI,
) -> dict:
    proposals = state.get("proposals", [])
    if not proposals:
        return {"proposals": []}
    collection = build_raw_collection(
        "practice_topics",
        embedding_model=settings.OPENAI_EMBEDDING_MODEL,
    )
    result = []
    accepted_embeddings: list[list[float]] = []
    for proposal in proposals:
        embedding = _embed(client, proposal.name)
        if _max_similarity(
            collection,
            embedding,
            where={"active": True},
        ) >= DUPLICATE_THRESHOLD:
            continue
        rejected_where = {
            "$and": [
                {"active": False},
                {"student_id": state["request"].student_id},
            ]
        }
        if _max_similarity(
            collection,
            embedding,
            where=rejected_where,
        ) >= DUPLICATE_THRESHOLD:
            continue
        if any(
            _cosine(embedding, previous) >= DUPLICATE_THRESHOLD
            for previous in accepted_embeddings
        ):
            continue
        result.append(proposal)
        accepted_embeddings.append(embedding)
    return {"proposals": result}


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
