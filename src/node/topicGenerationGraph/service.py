from openai import OpenAI

from config.chroma_config import settings
from node.topicGenerationGraph.graphConfig import TopicGenerationGraph
from schemas.topic_generation import (
    TopicIndexRequest,
    TopicProposalBatch,
    TopicProposalRequest,
    TopicSearchHit,
    TopicSearchRequest,
    TopicSearchResponse,
)
from vector.chroma_client import build_raw_collection

_graph: TopicGenerationGraph | None = None


def propose_topics(request: TopicProposalRequest) -> TopicProposalBatch:
    return _topic_graph().invoke(request)


def index_topic(request: TopicIndexRequest) -> None:
    client = OpenAI()
    text = f"{request.name}\n{request.description}".strip()
    response = client.embeddings.create(
        model=settings.OPENAI_EMBEDDING_MODEL,
        input=text,
    )
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
        embeddings=[response.data[0].embedding],
        documents=[text],
        metadatas=[metadata],
    )


def _topic_graph() -> TopicGenerationGraph:
    global _graph
    if _graph is None:
        _graph = TopicGenerationGraph()
    return _graph


def search_topics(request: TopicSearchRequest) -> TopicSearchResponse:
    """Tim chu de gan nghia voi tu khoa, tra ve ID kem do tuong dong.

    CHI tra ID: ten va mo ta trong Chroma la ban chup luc index, co the da cu hoac chu de da bi
    tat active. Java doc lai tu Postgres roi moi hien -- vector store dung de TIM, khong dung
    lam nguon hien thi.

    Chu de vat chat hoa tu ngan hang de cua truong (source = EXAM_QUESTION_BANK) khong di qua
    index_topic nen KHONG co o day. Khong sao: duong tim theo chuoi ben Java luon lay duoc chung.
    """
    client = OpenAI()
    embedding = client.embeddings.create(
        model=settings.OPENAI_EMBEDDING_MODEL,
        input=request.keyword.strip(),
    ).data[0].embedding

    collection = build_raw_collection(
        "practice_topics",
        embedding_model=settings.OPENAI_EMBEDDING_MODEL,
    )
    response = collection.query(
        query_embeddings=[embedding],
        n_results=request.limit,
        include=["distances"],
        where={"active": True},
    )
    ids = (response.get("ids") or [[]])[0]
    distances = (response.get("distances") or [[]])[0]

    hits: list[TopicSearchHit] = []
    for topic_id, distance in zip(ids, distances):
        # Chroma tra KHOANG CACH cosine; doi sang do tuong dong roi kep ve [0,1] vi sai so dau
        # phay dong co the cho ra 1.0000001 hoac -0.0000001.
        similarity = max(0.0, min(1.0, 1.0 - float(distance)))
        if similarity >= request.min_similarity:
            hits.append(TopicSearchHit(topic_id=str(topic_id), similarity=similarity))
    return TopicSearchResponse(hits=hits)
