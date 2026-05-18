"""Publish messages to RabbitMQ exchanges.

Messages are sent as **plain JSON** (no MassTransit envelope) because the
.NET consumer is configured with ``UseRawJsonDeserializer``.
"""

import logging

import aio_pika

from config.rabbitMq_config import settings
from infra.message_broker import connection
from infra.message_broker.models import PaperIngestionCompletedMessage

logger = logging.getLogger(__name__)


async def publish_paper_ingestion_completed(
    message: PaperIngestionCompletedMessage,
) -> None:
    """Publish a ``PaperIngestionCompletedEvent`` to RabbitMQ.

    The exchange is declared as *fanout* to match the MassTransit default
    topology.  The message body uses **camelCase** keys so .NET can
    deserialise it directly.
    """
    conn = await connection.connect()
    channel = await conn.channel()

    exchange = await channel.declare_exchange(
        settings.RABBITMQ_COMPLETED_EXCHANGE,
        type=aio_pika.ExchangeType.FANOUT,
        durable=True,
    )

    body = message.model_dump_json(by_alias=True).encode()

    await exchange.publish(
        aio_pika.Message(
            body=body,
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        ),
        routing_key="",
    )

    logger.info(
        "Published PaperIngestionCompletedEvent for paper %s (success=%s)",
        message.paper_id,
        message.is_success,
    )
