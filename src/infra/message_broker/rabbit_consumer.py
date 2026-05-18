import json
import logging

from vector.indexer import handle_outbox_event
from .connection import get_channel
from config.rabbitMq_config import settings

logger = logging.getLogger(__name__)

async def start_outbox_consumer(app):
    collection = app.state.chroma_collection

    channel = await get_channel()
    await channel.set_qos(prefetch_count=10)

    queue = await channel.declare_queue(settings.VECTOR_INDEX_QUEUE, durable=True)
    logger.warning("[outbox-consumer] LISTENING queue=%s", settings.VECTOR_INDEX_QUEUE)

    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            try:
                envelope = json.loads(message.body.decode())
                await handle_outbox_event(collection, envelope)
                await message.ack()
            except Exception:
                logger.exception("[outbox-consumer] FAILED → requeue")
                await message.nack(requeue=True)
