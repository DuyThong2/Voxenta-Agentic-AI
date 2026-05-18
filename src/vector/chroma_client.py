import json
import hashlib
import logging
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from config.chroma_config import settings

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _metadata_hash(md: dict) -> str:
    md_for_hash = dict(md)
    md_for_hash.pop("textHash", None)
    md_for_hash.pop("metadataHash", None)
    return hashlib.sha256(_stable_json(md_for_hash).encode("utf-8")).hexdigest()


def to_metadata(p: dict) -> dict:
    md = {
        "productId": str(p.get("productId") or ""),
        "productDetailId": str(p.get("productDetailId") or ""),
        "categoryId": str(p.get("categoryId") or ""),
        "categoryName": str(p.get("categoryName") or ""),
        "productDetailImage": str(p.get("productDetailImage") or ""),
        "productName": str(p.get("productName") or ""),
        "size": str(p.get("size") or ""),
        "colorFamily": str(p.get("colorFamily") or ""),
        "colorRaw": str(p.get("colorRaw") or ""),
        "price": float(p.get("price") or 0.0),
        "stock": int(p.get("stock") or 0),
        "status": str(p.get("status") or ""),
        "isSale": bool(p.get("isSale")),
        "updatedAt": str(p.get("updatedAt") or ""),
    }
    return md


def build_text(p: dict) -> str:
    text = (p.get("text") or "").strip()
    if text:
        return text
    return f"{p.get('productName','')} Size: {p.get('size','')}  Color: {p.get('colorRaw','')}".strip()


def log_model_with_metadata(p: dict, *, prefix: str = "[model]"):
    text = build_text(p)
    md = to_metadata(p)
    md["textHash"] = _text_hash(text)
    md["metadataHash"] = _metadata_hash(md)

    preview = text if len(text) <= 400 else text[:400] + "...(truncated)"
    logger.info("%s text=%r metadata=%s", prefix, preview, json.dumps(md, ensure_ascii=False))


# ── collection builder ───────────────────────────────────────────────────


def build_chroma_collection():
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable")

    client = chromadb.HttpClient(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        embedding_fn = OpenAIEmbeddingFunction(
            api_key=settings.OPENAI_API_KEY,
            model_name=settings.OPENAI_EMBEDDING_MODEL,
        )
    except TypeError:
        embedding_fn = OpenAIEmbeddingFunction(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_EMBEDDING_MODEL,
        )

    collection = client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION,
        embedding_function=embedding_fn,
    )

    logger.warning(
        "[chroma] connected host=%s port=%s collection=%s embed_provider=openai embed_model=%s",
        settings.CHROMA_HOST, settings.CHROMA_PORT, settings.CHROMA_COLLECTION, settings.OPENAI_EMBEDDING_MODEL,
    )
    return collection


# ── chroma operations ────────────────────────────────────────────────────


def upsert_to_chroma(collection, p: dict):
    doc_id = str(p.get("productDetailId") or "")
    if not doc_id:
        raise ValueError("payload missing productDetailId")

    text = build_text(p)
    md = to_metadata(p)
    md["textHash"] = _text_hash(text)
    md["metadataHash"] = _metadata_hash(md)

    try:
        existing = collection.get(ids=[doc_id], include=["metadatas", "documents"])
        existing_md = (existing.get("metadatas") or [None])[0] or {}
        existing_doc = (existing.get("documents") or [None])[0] or ""

        same_text = existing_md.get("textHash") == md["textHash"]
        same_metadata = existing_md.get("metadataHash") == md["metadataHash"]

        if same_text and same_metadata:
            logger.info("[chroma] skip unchanged id=%s", doc_id)
            return

        if same_text and not same_metadata:
            collection.upsert(ids=[doc_id], documents=[existing_doc or text], metadatas=[md])
            logger.info("[chroma] metadata refreshed id=%s", doc_id)
            return

    except Exception:
        logger.exception("[chroma] failed to inspect existing doc, fallback to upsert id=%s", doc_id)

    collection.upsert(ids=[doc_id], documents=[text], metadatas=[md])
    logger.info("[chroma] upserted+embedded id=%s", doc_id)


def delete_from_chroma(collection, doc_id: str):
    """Delete a single document by productDetailId."""
    if not doc_id:
        raise ValueError("doc_id is required")
    collection.delete(ids=[str(doc_id)])
    logger.info("[chroma] deleted id=%s", doc_id)


# ── demo ─────────────────────────────────────────────────────────────────

def demo_delete(doc_id: str = "test-product-detail-001"):
    collection = build_chroma_collection()
    delete_from_chroma(collection, doc_id)
    print(f"Deleted doc_id={doc_id}")
