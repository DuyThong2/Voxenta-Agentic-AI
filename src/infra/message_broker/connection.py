"""RabbitMQ connection manager using aio-pika.

Provides a singleton-style async connection that is created once during
application startup and closed on shutdown.
"""

import logging
from typing import Optional

import aio_pika
from aio_pika.abc import AbstractRobustConnection, AbstractChannel

from config.rabbitMq_config import settings

logger = logging.getLogger(__name__)

_connection: Optional[AbstractRobustConnection] = None
_channel: Optional[AbstractChannel] = None


async def connect() -> AbstractRobustConnection:
    """Establish (or return existing) robust connection to RabbitMQ."""
    global _connection
    if _connection is None or _connection.is_closed:
        logger.info("Connecting to RabbitMQ")
        _connection = await aio_pika.connect_robust(
            settings.RABBITMQ_URI,
            heartbeat=1200,  
        )
        logger.info("RabbitMQ connection established.")
    return _connection


async def get_channel() -> AbstractChannel:
    """Return a channel, creating one if necessary."""
    global _channel
    conn = await connect()
    if _channel is None or _channel.is_closed:
        _channel = await conn.channel()
        # Fair dispatch – one unacknowledged message at a time per consumer.
        await _channel.set_qos(prefetch_count=1)
    return _channel


async def close() -> None:
    """Gracefully close channel and connection."""
    global _channel, _connection
    if _channel and not _channel.is_closed:
        await _channel.close()
        _channel = None
    if _connection and not _connection.is_closed:
        await _connection.close()
        _connection = None
    logger.info("RabbitMQ connection closed.")
