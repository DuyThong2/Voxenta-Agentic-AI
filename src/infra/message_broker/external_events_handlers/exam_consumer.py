import asyncio
import json
import logging
import os

from config.kafka_config import settings
from events import ExamAttemptEvaluationRequestedEvent
from events.exam_attempt_evaluation_failed import (
    ExamAttemptEvaluationFailedEvent,
    ExamAttemptEvaluationFailedPayload,
)
from infra.message_broker.connection import get_topic_consumer
from infra.message_broker.publishers.exam_publisher import publish_exam_attempt_evaluation_failed
from infra.storage.audio_storage import download_from_s3
from node.state_models import QuestionContext, SpeakingInput, TopicContext
from schemas.enums import SpeakingMode

logger = logging.getLogger(__name__)


async def start_exam_attempt_consumer(app):
    consumer = await get_topic_consumer(
        settings.KAFKA_EXAM_REQUEST_TOPIC,
        group_id=settings.KAFKA_EXAM_CONSUMER_GROUP,
    )
    graph = app.state.graph

    async for message in consumer:
        payload: dict = {}
        retries = 0
        while retries <= settings.KAFKA_MAX_RETRY:
            try:
                payload = json.loads(message.value.decode())
                request_event = ExamAttemptEvaluationRequestedEvent.model_validate(payload)
                request_payload = request_event.payload
                main_turn = min(request_payload.turns, key=lambda turn: turn.turn_order)
                local_audio_path = download_from_s3(main_turn.audio_ref)
                initial_state = {
                    "speaking_input": SpeakingInput(
                        exam_attempt_id=request_event.exam_attempt_id,
                        answer_id=request_event.answer_id,
                        question_id=request_event.question_id,
                        audio_path=local_audio_path,
                        reference_text=request_payload.reference_text,
                        transcribed_text=main_turn.transcript,
                        mode=SpeakingMode(request_payload.mode),
                        language=request_payload.language,
                        criteria_frameworks=request_payload.criteria_frameworks,
                        question=QuestionContext(
                            question_text=request_payload.question_text,
                            question_type=request_payload.question_type,
                            difficulty_level=request_payload.difficulty_level,
                            duration_seconds=request_payload.duration_seconds,
                            min_response_seconds=request_payload.min_response_seconds,
                            max_response_seconds=request_payload.max_response_seconds,
                            evaluation_guide=request_payload.evaluation_guide,
                        ),
                        topic=TopicContext(
                            topic_name=request_payload.topic_name,
                            topic_description=request_payload.topic_description,
                        ),
                    ),
                    "status": "idle",
                    "metadata": {"request_payload": payload},
                }
                try:
                    await asyncio.to_thread(
                        graph.invoke,
                        initial_state,
                        {"configurable": {"thread_id": f"{request_event.exam_attempt_id}:{request_event.answer_id}"}},
                    )
                finally:
                    if local_audio_path != main_turn.audio_ref and os.path.exists(local_audio_path):
                        os.unlink(local_audio_path)
                await consumer.commit()
                break
            except Exception as exc:
                retries += 1
                logger.exception(
                    "[exam-consumer] failed topic=%s partition=%s offset=%s retry=%s",
                    message.topic,
                    message.partition,
                    message.offset,
                    retries,
                )
                if retries > settings.KAFKA_MAX_RETRY:
                    await publish_exam_attempt_evaluation_failed(
                        ExamAttemptEvaluationFailedEvent(
                            exam_attempt_id=payload.get("examAttemptId", "unknown"),
                            answer_id=payload.get("answerId", "unknown"),
                            question_id=payload.get("questionId", "unknown"),
                            payload=ExamAttemptEvaluationFailedPayload(
                                error=str(exc),
                                retry_count=retries,
                            ),
                        )
                    )
                    await consumer.commit()
                    break
