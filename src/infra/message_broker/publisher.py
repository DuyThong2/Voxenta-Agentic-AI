"""Publish messages to Kafka topics as plain JSON."""

import logging

from config.kafka_config import settings
from infra.message_broker import connection
from infra.message_broker.models import PaperIngestionCompletedMessage

logger = logging.getLogger(__name__)


async def publish_paper_ingestion_completed(
    message: PaperIngestionCompletedMessage,
) -> None:
    """Publish a ``PaperIngestionCompletedEvent`` to Kafka."""
    producer = await connection.get_producer()

    body = message.model_dump_json(by_alias=True).encode()

    await producer.send_and_wait(
        settings.KAFKA_COMPLETED_TOPIC,
        body,
        key=message.paper_id.encode(),
    )

    logger.info(
        "Published PaperIngestionCompletedEvent topic=%s paper=%s success=%s",
        settings.KAFKA_COMPLETED_TOPIC,
        message.paper_id,
        message.is_success,
    )
