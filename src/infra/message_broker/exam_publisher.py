import logging

from config.kafka_config import settings
from infra.message_broker.events import (
    ExamAttemptEvaluationCompletedEvent,
    ExamAttemptEvaluationFailedEvent,
)
from infra.message_broker import connection

logger = logging.getLogger(__name__)


async def _publish(event, *, answer_id: str) -> None:
    producer = await connection.get_producer()
    body = event.model_dump_json(by_alias=True).encode()

    await producer.send_and_wait(
        settings.KAFKA_EXAM_COMPLETED_TOPIC,
        body,
        key=answer_id.encode(),
    )

    logger.info(
        "Published %s topic=%s answer=%s",
        event.event_type,
        settings.KAFKA_EXAM_COMPLETED_TOPIC,
        answer_id,
    )


async def publish_exam_attempt_evaluation_completed(
    event: ExamAttemptEvaluationCompletedEvent,
) -> None:
    await _publish(event, answer_id=event.answer_id)


async def publish_exam_attempt_evaluation_failed(
    event: ExamAttemptEvaluationFailedEvent,
) -> None:
    await _publish(event, answer_id=event.answer_id)
