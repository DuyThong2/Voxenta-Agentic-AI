from openai import OpenAI

from config.chroma_config import settings
from node.topicGenerationGraph.graphConfig import TopicGenerationGraph
from schemas.topic_generation import (
    TopicIndexRequest,
    TopicProposalBatch,
    TopicProposalRequest,
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
