import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Tuple

from config.kafka_config import settings
from events import ExamAttemptEvaluationRequestedEvent
from events.exam_attempt_evaluation_completed import (
    ExamAttemptEvaluationCompletedEvent,
    ExamAttemptEvaluationCompletedPayload,
)
from events.exam_attempt_evaluation_failed import (
    ExamAttemptEvaluationFailedEvent,
    ExamAttemptEvaluationFailedPayload,
)
from infra.message_broker.connection import get_topic_consumer
from infra.message_broker.publishers.exam_publisher import (
    publish_exam_attempt_evaluation_completed,
    publish_exam_attempt_evaluation_failed,
)
from infra.storage.audio_storage import download_from_s3_async
from mappers.exam_event_builder import (
    build_completed_event,
    build_criteria_payload,
    build_feedback_payload,
    build_signals,
    build_turn_detail,
)
from node.evalGraph.AnswerLengthNode.answer_length_analysis_node_config import (
    answer_length_analysis_node,
)
from node.evalGraph.CoherenceEvalNode.coherence_eval_node_config import coherence_eval_node
from node.evalGraph.GrammarEvalNode.grammar_eval_node_config import grammar_eval_node
from node.evalGraph.LexicalEvalNode.lexical_eval_node_config import lexical_eval_node
from node.state_models import QuestionAssetContext, QuestionContext, SpeakingInput, TopicContext
from node.state_models.pronunciation import FormattedPronunciationResult
from schemas.enums import SpeakingMode

logger = logging.getLogger(__name__)


def _real_transcript_for_turn(result: Dict[str, Any]) -> str:
    """The turn's actual transcribed text, from start_node's own Azure transcription.

    turn.transcript (the raw field on the Kafka request payload, coming from vox) is
    never populated -- vox's archive flow never transcribes anything itself, only
    Python's start_node does, per-turn, at graph.invoke() time. Anything that wants
    "what did the student actually say for this turn" must read it from here
    (result["speaking_input"].transcribed_text), not from turn.transcript.
    """
    speaking_input = result.get("speaking_input")
    if speaking_input is None:
        return ""
    return speaking_input.transcribed_text or speaking_input.corrected_transcript or ""


def _combine_transcript(per_turn_results: List[Tuple[Any, Dict[str, Any]]]) -> str:
    """User-only merged transcript (no AI lines, no timestamps).

    This is what grammar/lexical/answer-length grade against -- the whole
    multi-turn answer, but only the student's own words, never the AI's
    prompt/follow-up text (which would otherwise inflate word/sentence counts).
    """
    lines: List[str] = []
    for _turn, result in sorted(per_turn_results, key=lambda item: item[0].turn_order):
        text = _real_transcript_for_turn(result).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _format_elapsed(total_seconds: int) -> str:
    total_seconds = max(0, total_seconds)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def _build_dialogue_transcript(per_turn_results: List[Tuple[Any, Dict[str, Any]]]) -> str:
    """Timestamped AI/User dialogue transcript, e.g.:
        [00:00] AI: What is your favorite color?
        [00:05] User: I like blue because the sky is blue.

    Used only as additional context for CoherenceEvalNode (to judge whether
    the answer coherently follows what was actually asked, including
    follow-ups) -- never fed into grammar/lexical/answer-length, which must
    only see the student's own words (see _combine_transcript above).
    Timestamps are approximate: each AI line inherits the running elapsed
    time, then the elapsed time advances by that turn's own duration_seconds
    after the User line -- ordering/pacing matters here, not precision.
    """
    lines: List[str] = []
    elapsed = 0
    for turn, result in sorted(per_turn_results, key=lambda item: item[0].turn_order):
        if turn.prompt_text and turn.prompt_text.strip():
            lines.append(f"[{_format_elapsed(elapsed)}] AI: {turn.prompt_text.strip()}")
        text = _real_transcript_for_turn(result).strip()
        if text:
            lines.append(f"[{_format_elapsed(elapsed)}] User: {text}")
        elapsed += int(turn.duration_seconds or 0)
    return "\n".join(lines)


def _total_duration_seconds(turns: List[Any]) -> int:
    return sum(int(turn.duration_seconds or 0) for turn in turns)


def _average_scores(values: List[float]) -> float:
    return round(sum(values) / len(values), 2)


def _build_question_context(request_payload: Any) -> QuestionContext:
    return QuestionContext(
        question_text=request_payload.question_text,
        question_type=request_payload.question_type,
        difficulty_level=request_payload.difficulty_level,
        duration_seconds=request_payload.duration_seconds,
        min_response_seconds=request_payload.min_response_seconds,
        max_response_seconds=request_payload.max_response_seconds,
        evaluation_guide=request_payload.evaluation_guide,
        asset=(
            QuestionAssetContext(
                type=request_payload.asset.type,
                transcript=request_payload.asset.transcript,
                description=request_payload.asset.description,
                alt_text=request_payload.asset.alt_text,
            )
            if getattr(request_payload, "asset", None) is not None
            else None
        ),
    )


def _build_topic_context(request_payload: Any) -> TopicContext:
    return TopicContext(
        topic_name=request_payload.topic_name,
        topic_description=request_payload.topic_description,
    )


def _build_initial_state(
    request_event: ExamAttemptEvaluationRequestedEvent,
    turn: Any,
    local_audio_path: str,
    conversation_transcript: str,
    raw_payload: Dict[str, Any],
) -> Dict[str, Any]:
    request_payload = request_event.payload
    return {
        "speaking_input": SpeakingInput(
            exam_attempt_id=request_event.exam_attempt_id,
            answer_id=request_event.answer_id,
            question_id=request_event.question_id,
            audio_path=local_audio_path,
            reference_text=request_payload.reference_text,
            transcribed_text=turn.transcript,
            conversation_transcript=conversation_transcript or turn.transcript,
            mode=SpeakingMode(request_payload.mode),
            language=request_payload.language,
            criteria_frameworks=request_payload.criteria_frameworks,
            question=_build_question_context(request_payload),
            topic=_build_topic_context(request_payload),
        ),
        "status": "idle",
        "metadata": {
            "request_payload": raw_payload,
            "turn_order": turn.turn_order,
        },
    }


async def _evaluate_turn(
    graph: Any,
    request_event: ExamAttemptEvaluationRequestedEvent,
    turn: Any,
    conversation_transcript: str,
    raw_payload: Dict[str, Any],
) -> Dict[str, Any]:
    local_audio_path = await download_from_s3_async(turn.audio_ref)
    try:
        initial_state = _build_initial_state(
            request_event,
            turn,
            local_audio_path,
            conversation_transcript,
            raw_payload,
        )
        result = await asyncio.to_thread(
            graph.invoke,
            initial_state,
            {
                "configurable": {
                    "thread_id": f"{request_event.exam_attempt_id}:{request_event.answer_id}:{turn.turn_order}",
                }
            },
        )
        if result.get("status") == "error":
            raise RuntimeError(result.get("error") or "evaluation graph returned error")
        return result
    finally:
        if local_audio_path != turn.audio_ref and os.path.exists(local_audio_path):
            os.unlink(local_audio_path)


def _run_aggregate_text_evaluation(
    per_turn_results: List[Tuple[Any, Dict[str, Any]]],
    merged_transcript: str,
    dialogue_transcript: str,
) -> Dict[str, Any]:
    """Grade grammar/lexical/coherence/answer-length ONCE, against the whole
    multi-turn answer -- not per turn.

    start_node (run inside each per-turn graph.invoke above) unconditionally
    re-transcribes speaking_input.transcribed_text from that turn's own
    audio_path, so there is no way to get a full-answer merged transcript
    through the compiled graph's normal entry point. Pronunciation/validity/
    correction are all inherently per-turn/per-audio and already ran once per
    turn via graph.invoke above; this function instead calls the remaining
    text-only node functions directly or, bypassing start_node/validity_node/
    correction_node/pronunciation_eval_node entirely, on a synthetic state
    whose transcribed_text is the merged, student-only transcript.
    """
    base_speaking_input = per_turn_results[0][1]["speaking_input"]
    speaking_input = base_speaking_input.model_copy(update={
        "transcribed_text": merged_transcript,
        "conversation_transcript": dialogue_transcript,
        "answer_length_metrics": None,
    })
    # A turn whose validity_node rejected it (action=reject_or_zero) routes straight to
    # END without ever reaching pronunciation_eval_node, so "pronunciation_result" is
    # absent from that turn's graph result -- fall back to the first turn that actually
    # has one, or a blank default if every turn in this answer was rejected.
    pronunciation_source = next(
        (result.get("pronunciation_result") for _turn, result in per_turn_results if result.get("pronunciation_result") is not None),
        None,
    )
    pronunciation_result = (
        pronunciation_source.model_copy(deep=True)
        if pronunciation_source is not None
        else FormattedPronunciationResult()
    )

    state: Dict[str, Any] = {
        "speaking_input": speaking_input,
        "pronunciation_result": pronunciation_result,
        "status": "processing",
        "error": None,
        "metadata": {},
        "validity": per_turn_results[0][1].get("validity"),
    }

    for node in (answer_length_analysis_node, coherence_eval_node, lexical_eval_node, grammar_eval_node):
        state = node(state)
        if state.get("status") == "error":
            raise RuntimeError(state.get("error") or f"{node.__name__} failed during aggregate text evaluation")

    return state


def _merge_multi_turn_criteria(
    aggregate_result: Dict[str, Any],
    per_turn_results: List[Tuple[Any, Dict[str, Any]]],
) -> Dict[str, Any]:
    criteria = build_criteria_payload(aggregate_result)
    if not criteria:
        return criteria

    pronunciation_scores: List[float] = []
    fluency_scores: List[float] = []
    for _turn, result in per_turn_results:
        per_turn_criteria = build_criteria_payload(result)
        pronunciation = per_turn_criteria.get("pronunciation")
        fluency = per_turn_criteria.get("fluency")
        if pronunciation is not None and pronunciation.score is not None:
            pronunciation_scores.append(pronunciation.score)
        if fluency is not None and fluency.score is not None:
            fluency_scores.append(fluency.score)

    if pronunciation_scores and criteria.get("pronunciation") is not None:
        criteria["pronunciation"] = criteria["pronunciation"].model_copy(update={
            "score": _average_scores(pronunciation_scores),
            "note": f"Averaged across {len(pronunciation_scores)} recorded turns.",
        })
    if fluency_scores and criteria.get("fluency") is not None:
        criteria["fluency"] = criteria["fluency"].model_copy(update={
            "score": _average_scores(fluency_scores),
            "note": f"Averaged across {len(fluency_scores)} recorded turns.",
        })
    return criteria


def _build_multi_turn_completed_event(
    request_event: ExamAttemptEvaluationRequestedEvent,
    aggregate_result: Dict[str, Any],
    aggregate_audio_path: str,
    per_turn_results: List[Tuple[Any, Dict[str, Any]]],
    total_duration_seconds: int,
) -> ExamAttemptEvaluationCompletedEvent:
    speaking_input = aggregate_result.get("speaking_input")
    feedback_summary, suggestions = build_feedback_payload(
        aggregate_result,
        speaking_input,
        audio_path=aggregate_audio_path,
    )
    turns = [
        build_turn_detail(
            result,
            turn_order=turn.turn_order,
            turn_type=turn.turn_type or "MAIN",
            prompt_text=turn.prompt_text,
            audio_url=turn.audio_ref,
            transcript=turn.transcript,
            duration_seconds=turn.duration_seconds,
        )
        for turn, result in per_turn_results
    ]

    return ExamAttemptEvaluationCompletedEvent(
        exam_attempt_id=request_event.exam_attempt_id,
        answer_id=request_event.answer_id,
        question_id=request_event.question_id,
        payload=ExamAttemptEvaluationCompletedPayload(
            turns=turns,
            criteria=_merge_multi_turn_criteria(aggregate_result, per_turn_results),
            signals=build_signals(aggregate_result, speaking_input, duration_seconds=total_duration_seconds),
            validity=aggregate_result.get("validity"),
            feedback_summary=feedback_summary,
            suggestions=suggestions,
            model_version="gpt-4o",
            prompt_version="v1",
            evaluated_at=build_completed_event(
                aggregate_result,
                speaking_input,
                audio_path=aggregate_audio_path,
            ).payload.evaluated_at,
        ),
    )


async def start_exam_attempt_consumer(app):
    consumer = await get_topic_consumer(
        settings.KAFKA_EXAM_REQUEST_TOPIC,
        group_id=settings.KAFKA_EXAM_CONSUMER_GROUP,
    )
    graph = app.state.graph

    async for message in consumer:
        payload: Dict[str, Any] = {}
        retries = 0
        while retries <= settings.KAFKA_MAX_RETRY:
            try:
                payload = json.loads(message.value.decode())
                request_event = ExamAttemptEvaluationRequestedEvent.model_validate(payload)
                request_payload = request_event.payload
                turns = sorted(request_payload.turns, key=lambda turn: turn.turn_order)
                if not turns:
                    raise ValueError("exam evaluation request does not contain any turns")

                logger.info(
                    "[exam-consumer] received answer_id=%s exam_attempt_id=%s turns=%d",
                    request_event.answer_id, request_event.exam_attempt_id, len(turns),
                )

                per_turn_results: List[Tuple[Any, Dict[str, Any]]] = []
                aggregate_audio_path = turns[0].audio_ref

                for turn in turns:
                    logger.info(
                        "[exam-consumer] evaluating turn %d/%d answer_id=%s",
                        turn.turn_order, len(turns), request_event.answer_id,
                    )
                    # No merged dialogue_transcript is passed in here -- turn.transcript
                    # (vox's own record) is never populated, so nothing meaningful could be
                    # built yet anyway. Real transcripts only exist after start_node runs,
                    # per turn, below. Whatever per-turn coherence_eval sees during this
                    # graph.invoke() is discarded/overridden by the aggregate step further
                    # down, which rebuilds an accurate merged/dialogue transcript from the
                    # real per-turn transcriptions once they're all known.
                    result = await _evaluate_turn(
                        graph,
                        request_event,
                        turn,
                        "",
                        payload,
                    )
                    per_turn_results.append((turn, result))
                    logger.info(
                        "[exam-consumer] turn %d/%d done answer_id=%s",
                        turn.turn_order, len(turns), request_event.answer_id,
                    )

                merged_transcript = _combine_transcript(per_turn_results)
                dialogue_transcript = _build_dialogue_transcript(per_turn_results)

                logger.info("[exam-consumer] running aggregate text evaluation answer_id=%s", request_event.answer_id)
                aggregate_result = await asyncio.to_thread(
                    _run_aggregate_text_evaluation,
                    per_turn_results,
                    merged_transcript,
                    dialogue_transcript,
                )
                completed_event = _build_multi_turn_completed_event(
                    request_event,
                    aggregate_result,
                    aggregate_audio_path,
                    per_turn_results,
                    _total_duration_seconds(turns),
                )
                await publish_exam_attempt_evaluation_completed(completed_event)
                await consumer.commit()
                logger.info("[exam-consumer] completed and published answer_id=%s", request_event.answer_id)
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
