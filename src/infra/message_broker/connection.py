import asyncio
import logging
from typing import Dict, Optional

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError, KafkaError

from config.kafka_config import settings

logger = logging.getLogger(__name__)

_producer: Optional[AIOKafkaProducer] = None
_consumers: Dict[str, AIOKafkaConsumer] = {}

# consumer.start() only tries the broker once and raises on failure -- with no
# retry, a Kafka pod that's mid-restart exactly when this pod boots (a real,
# recurring race: Karpenter node churn, Kafka rollouts, etc.) permanently
# kills that consumer's asyncio.create_task() with no trace anywhere (nothing
# ever awaits it), confirmed for real: exam-attempt-evaluation-requested built
# up 4 unconsumed messages for ~45 minutes after one such race, invisible
# until someone manually checked `kafka-consumer-groups.sh` and found the
# group didn't even exist on the broker.
async def _start_with_retry(consumer: AIOKafkaConsumer, *, label: str) -> None:
    delay = 2
    while True:
        try:
            await consumer.start()
            return
        except (KafkaConnectionError, KafkaError) as exc:
            logger.error(
                "Kafka consumer failed to start (%s), retrying in %ss: %s",
                label, delay, exc,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)


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
        await _start_with_retry(consumer, label=f"topic={topic}")
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
        await _start_with_retry(consumer, label=f"topic={topic} group={group_id}")
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
