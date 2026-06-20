import logging
from typing import Dict, Optional

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from config.kafka_config import settings

logger = logging.getLogger(__name__)

_producer: Optional[AIOKafkaProducer] = None
_consumers: Dict[str, AIOKafkaConsumer] = {}


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
    """Return a started Kafka consumer singleton scoped by topic."""
    consumer = _consumers.get(topic)
    if consumer is None:
        logger.info("Connecting Kafka consumer topic=%s", topic)
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            client_id=settings.KAFKA_CLIENT_ID,
            group_id=settings.KAFKA_CONSUMER_GROUP,
            enable_auto_commit=False,
            auto_offset_reset=settings.KAFKA_AUTO_OFFSET_RESET,
        )
        await consumer.start()
        _consumers[topic] = consumer
        logger.info("Kafka consumer established.")
    return consumer


async def get_topic_consumer(topic: str, *, group_id: str) -> AIOKafkaConsumer:
    cache_key = f"{group_id}:{topic}"
    consumer = _consumers.get(cache_key)
    if consumer is None:
        logger.info("Connecting Kafka consumer topic=%s group=%s", topic, group_id)
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            client_id=settings.KAFKA_CLIENT_ID,
            group_id=group_id,
            enable_auto_commit=False,
            auto_offset_reset=settings.KAFKA_AUTO_OFFSET_RESET,
        )
        await consumer.start()
        _consumers[cache_key] = consumer
    return consumer


async def close() -> None:
    """Gracefully close Kafka clients."""
    global _producer
    for consumer in _consumers.values():
        await consumer.stop()
    _consumers.clear()
    if _producer is not None:
        await _producer.stop()
        _producer = None
    logger.info("Kafka clients closed.")
