import json
import logging

from config.kafka_config import settings
from events import (
    QuestionAssetAnalysisCompletedEvent,
    QuestionAssetAnalysisCompletedPayload,
    QuestionAssetAnalysisRequestedEvent,
)
from infra.message_broker.connection import get_topic_consumer
from infra.message_broker.publishers.exam_publisher import (
    publish_question_asset_analysis_completed,
)
from node.assetAnalysisGraph import analyze_asset_request

logger = logging.getLogger(__name__)


async def start_question_asset_analysis_consumer(app):
    consumer = await get_topic_consumer(
        settings.KAFKA_QUESTION_ASSET_ANALYSIS_REQUEST_TOPIC,
        group_id=settings.KAFKA_QUESTION_ASSET_ANALYSIS_CONSUMER_GROUP,
    )

    async for message in consumer:
        try:
            payload = json.loads(message.value.decode())
            event = QuestionAssetAnalysisRequestedEvent.model_validate(payload)
            result = await __import__("asyncio").to_thread(analyze_asset_request, event)
            await publish_question_asset_analysis_completed(
                QuestionAssetAnalysisCompletedEvent(
                    asset_id=event.asset_id,
                    payload=QuestionAssetAnalysisCompletedPayload(
                        transcript=result.transcript,
                        description=result.description,
                    ),
                )
            )
        except Exception:
            logger.exception(
                "[question-asset-analysis] failed topic=%s partition=%s offset=%s",
                message.topic,
                message.partition,
                message.offset,
            )
        finally:
            await consumer.commit()
