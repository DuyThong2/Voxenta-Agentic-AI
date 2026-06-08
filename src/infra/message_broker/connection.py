"""Kafka connection manager for async producer and consumer clients."""

import logging
from typing import Optional

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from config.kafka_config import settings

logger = logging.getLogger(__name__)

_producer: Optional[AIOKafkaProducer] = None
_consumer: Optional[AIOKafkaConsumer] = None


async def get_producer() -> AIOKafkaProducer:
    """Return a started Kafka producer singleton."""
    global _producer
    if _producer is None:
        logger.info("Connecting Kafka producer")
        _producer = AIOKafkaProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            client_id=settings.KAFKA_CLIENT_ID,
            acks="all",
        )
        await _producer.start()
        logger.info("Kafka producer established.")
    return _producer


async def get_consumer(topic: str) -> AIOKafkaConsumer:
    """Return a started Kafka consumer singleton for the given topic."""
    global _consumer
    if _consumer is None:
        logger.info("Connecting Kafka consumer topic=%s", topic)
        _consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            client_id=settings.KAFKA_CLIENT_ID,
            group_id=settings.KAFKA_CONSUMER_GROUP,
            enable_auto_commit=False,
            auto_offset_reset=settings.KAFKA_AUTO_OFFSET_RESET,
        )
        await _consumer.start()
        logger.info("Kafka consumer established.")
    return _consumer


async def close() -> None:
    """Gracefully close Kafka clients."""
    global _producer, _consumer
    if _consumer is not None:
        await _consumer.stop()
        _consumer = None
    if _producer is not None:
        await _producer.stop()
        _producer = None
    logger.info("Kafka clients closed.")
