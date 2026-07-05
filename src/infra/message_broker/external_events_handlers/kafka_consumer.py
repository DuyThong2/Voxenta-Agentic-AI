import json
import logging
from asyncio import sleep

from aiokafka import TopicPartition

from config.kafka_config import settings
from infra.message_broker.connection import get_consumer
from vector.indexer import handle_outbox_event

logger = logging.getLogger(__name__)


async def start_outbox_consumer(app):
    collection = app.state.chroma_collection
    consumer = await get_consumer(settings.KAFKA_VECTOR_INDEX_TOPIC)

    logger.warning(
        "[outbox-consumer] LISTENING topic=%s group=%s",
        settings.KAFKA_VECTOR_INDEX_TOPIC,
        settings.KAFKA_CONSUMER_GROUP,
    )

    async for message in consumer:
        try:
            envelope = json.loads(message.value.decode())
            await handle_outbox_event(collection, envelope)
            await consumer.commit()
        except Exception:
            logger.exception(
                "[outbox-consumer] FAILED topic=%s partition=%s offset=%s",
                message.topic,
                message.partition,
                message.offset,
            )
            consumer.seek(
                TopicPartition(message.topic, message.partition),
                message.offset,
            )
            await sleep(1)
