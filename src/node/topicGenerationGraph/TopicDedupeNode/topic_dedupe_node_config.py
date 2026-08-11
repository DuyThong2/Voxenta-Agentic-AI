import logging

from openai import OpenAI

from config.chroma_config import settings
from node.topicGenerationGraph.constants import DUPLICATE_THRESHOLD
from node.topicGenerationGraph.GraphState import TopicGenerationState
from vector.chroma_client import build_raw_collection

logger = logging.getLogger(__name__)


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
    # Tich luy qua cac vong: vong sau phai so duoc voi cai vong truoc da nhan, vi chung CHUA
    # nam trong Chroma (Java moi index sau khi tao xong).
    result = list(state.get("accepted") or [])
    accepted_embeddings: list[list[float]] = list(state.get("accepted_embeddings") or [])
    dropped: list[str] = []
    collisions: list[str] = []
    for proposal in proposals:
        embedding = _embed(client, proposal.name)
        similarity, collided_name = _max_similarity(
            collection,
            embedding,
            where={"active": True},
        )
        if similarity >= DUPLICATE_THRESHOLD:
            dropped.append(f"{proposal.name} ~ {collided_name or '?'} ({similarity:.2f}, kho)")
            if collided_name:
                collisions.append(collided_name)
            continue
        rejected_where = {
            "$and": [
                {"active": False},
                {"student_id": state["request"].student_id},
            ]
        }
        similarity, collided_name = _max_similarity(
            collection,
            embedding,
            where=rejected_where,
        )
        if similarity >= DUPLICATE_THRESHOLD:
            dropped.append(f"{proposal.name} ~ {collided_name or '?'} ({similarity:.2f}, da-loai)")
            if collided_name:
                collisions.append(collided_name)
            continue
        if any(
            _cosine(embedding, previous) >= DUPLICATE_THRESHOLD
            for previous in accepted_embeddings
        ):
            dropped.append(f"{proposal.name} ~ (trung trong cung lo)")
            continue
        result.append(proposal)
        accepted_embeddings.append(embedding)
    logger.info(
        "[topic-dedupe] vong=%s nhan=%d giu_tich_luy=%d bo=%d%s",
        state.get("round"), len(proposals), len(result), len(dropped),
        "" if not dropped else " | " + " ; ".join(dropped),
    )
    return {
        "accepted": result,
        "accepted_embeddings": accepted_embeddings,
        # Bo trung trong chinh danh sach va cham -- hai de xuat dam vao cung mot chu de thi khong
        # can noi hai lan, va giu tran token cho vong sau.
        "collisions": list(dict.fromkeys(collisions)),
    }


def _max_similarity(
    collection, embedding: list[float], *, where: dict
) -> tuple[float, str | None]:
    """Do tuong dong cao nhat, KEM TEN chu de gay ra no.

    Tra thêm ten vi bo loc nay dang vut de xuat di trong im lang -- khong ai biet mot luot sinh
    ra 0 chu de la do "kho da du" hay do "8 de xuat deu trung". Ten cai va cham chinh la thu tra
    loi cau do, va cung la thu can de sau nay dua nguoc vao prompt cho vong de xuat lai.
    """
    if collection.count() == 0:
        return 0.0, None
    response = collection.query(
        query_embeddings=[embedding],
        n_results=1,
        where=where,
        include=["distances", "metadatas"],
    )
    distances = response.get("distances") or []
    if not distances or not distances[0]:
        return 0.0, None
    similarity = max(0.0, min(1.0, 1.0 - float(distances[0][0])))
    metadatas = response.get("metadatas") or []
    name = None
    if metadatas and metadatas[0]:
        name = (metadatas[0][0] or {}).get("name")
    return similarity, name


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
