import logging
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from config.chroma_config import settings

logger = logging.getLogger(__name__)


def _build_openai_embedding_function(model: str):
    try:
        return OpenAIEmbeddingFunction(api_key=settings.OPENAI_API_KEY, model_name=model)
    except TypeError:
        return OpenAIEmbeddingFunction(api_key=settings.OPENAI_API_KEY, model=model)


def build_raw_collection(name: str, *, embedding_model: str):
    """Build/get a Vox practice-question/topic collection, embedded via OpenAI.

    Guards against silently mixing embedding models within a collection: the
    HNSW index is fixed-dimension, so switching models on an existing
    collection requires dropping and recreating it, not just changing this
    call site.
    """
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable")

    client = chromadb.HttpClient(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=name,
        embedding_function=_build_openai_embedding_function(embedding_model),
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": embedding_model,
        },
    )
    configured_model = (collection.metadata or {}).get("embedding_model")
    if configured_model != embedding_model:
        raise RuntimeError(
            f"Collection {name} uses {configured_model}; "
            f"changing to {embedding_model} requires a vector migration"
        )
    return collection
