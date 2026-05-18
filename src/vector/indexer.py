import json
import logging

from vector.chroma_client import (
    log_model_with_metadata,
    upsert_to_chroma,
    delete_from_chroma,
)

logger = logging.getLogger(__name__)


def unwrap_envelope(envelope: dict) -> dict:
    if isinstance(envelope, dict) and isinstance(envelope.get("message"), dict):
        return envelope["message"]
    return envelope if isinstance(envelope, dict) else {}


def parse_payload(data: dict) -> dict:
    payload = data.get("payload")
    if isinstance(payload, str):
        return json.loads(payload)
    return payload or {}


async def handle_outbox_event(collection, envelope: dict):
    data = unwrap_envelope(envelope)
    event_type = data.get("eventType")
    payload = parse_payload(data)

    logger.info(
        "[outbox] eventType=%s aggregateType=%s aggregateId=%s outboxEventId=%s",
        event_type,
        data.get("aggregateType"),
        data.get("aggregateId"),
        data.get("outboxEventId"),
    )

    log_model_with_metadata(payload, prefix=f"[incoming:{event_type}]")

    if event_type == "ProductDetail.Upserted":
        upsert_to_chroma(collection, payload)
        return

    if event_type == "ProductDetail.Deleted":
        delete_from_chroma(collection, str(payload.get("productDetailId", "")))
        return

    logger.info("[consumer] skip eventType=%s", event_type)
