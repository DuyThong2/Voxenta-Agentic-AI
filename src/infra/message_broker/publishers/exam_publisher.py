import json
import logging

from config.kafka_config import settings
from events import (
    AnswerTurnsRecordedEvent,
    ExamAttemptEvaluationCompletedEvent,
    ExamAttemptEvaluationFailedEvent,
    QuestionAssetAnalysisCompletedEvent,
)
from infra.message_broker import connection

logger = logging.getLogger(__name__)


async def _publish(event, *, answer_id: str, topic: str) -> None:
    producer = await connection.get_producer()
    body = event.model_dump_json(by_alias=True).encode()

    await producer.send_and_wait(
        topic,
        body,
        key=answer_id.encode(),
    )

    logger.info(
        "Published %s topic=%s answer=%s",
        event.event_type,
        topic,
        answer_id,
    )


async def publish_exam_attempt_evaluation_completed(
    event: ExamAttemptEvaluationCompletedEvent,
) -> None:
    await _publish(
        event,
        answer_id=event.answer_id,
        topic=settings.KAFKA_EXAM_COMPLETED_TOPIC,
    )


async def publish_exam_attempt_evaluation_failed(
    event: ExamAttemptEvaluationFailedEvent,
) -> None:
    await _publish(
        event,
        answer_id=event.answer_id,
        topic=settings.KAFKA_EXAM_COMPLETED_TOPIC,
    )


async def publish_practice_attempt_evaluation_completed(
    event: ExamAttemptEvaluationCompletedEvent,
) -> None:
    producer = await connection.get_producer()
    body, session_id = practice_event_body(
        event,
        "PracticeAttemptEvaluationCompleted",
    )
    await producer.send_and_wait(
        settings.KAFKA_PRACTICE_COMPLETED_TOPIC,
        json.dumps(body).encode(),
        key=session_id.encode(),
    )


async def publish_practice_attempt_evaluation_failed(
    event: ExamAttemptEvaluationFailedEvent,
) -> None:
    producer = await connection.get_producer()
    body, session_id = practice_event_body(
        event,
        "PracticeAttemptEvaluationFailed",
    )
    await producer.send_and_wait(
        settings.KAFKA_PRACTICE_COMPLETED_TOPIC,
        json.dumps(body).encode(),
        key=session_id.encode(),
    )


def practice_event_body(event, event_type: str) -> tuple[dict, str]:
    body = event.model_dump(by_alias=True)
    session_id = body.pop("examAttemptId")
    body["practiceSessionId"] = session_id
    body["practiceResponseId"] = body.pop("answerId")
    body["practiceQuestionId"] = body.pop("questionId")
    body["eventType"] = event_type
    return body, session_id


async def publish_answer_turns_recorded(
    event: AnswerTurnsRecordedEvent,
) -> None:
    await _publish(
        event,
        answer_id=event.answer_id,
        topic=settings.KAFKA_ANSWER_TURN_TOPIC,
    )


async def publish_question_asset_analysis_completed(
    event: QuestionAssetAnalysisCompletedEvent,
) -> None:
    await _publish(
        event,
        answer_id=event.asset_id,
        topic=settings.KAFKA_QUESTION_ASSET_ANALYSIS_COMPLETED_TOPIC,
    )
