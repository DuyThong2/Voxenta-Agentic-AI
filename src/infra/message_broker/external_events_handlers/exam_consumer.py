import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

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
from events.exam_attempt_evaluation_shared import EvaluationSignals
from events.exam_attempt_force_end_requested import ExamAttemptForceEndRequestedEvent
from infra.message_broker.connection import (
    get_topic_consumer,
    get_topic_consumer_instance,
)
from infra.message_broker.publishers.exam_publisher import (
    publish_exam_attempt_evaluation_completed,
    publish_exam_attempt_evaluation_failed,
)
from infra.storage.audio_storage import download_from_s3_async
from schemas.scoring import CriterionScore
from schemas.validity import RuleResult, ValidityResult
from mappers.exam_event_builder import (
    build_completed_event,
    build_criteria_payload,
    build_feedback_payload,
    build_signals,
    build_turn_detail,
    display_transcript,
)
from node.evalGraph.AnswerLengthNode.answer_length_analysis_node_config import (
    answer_length_analysis_node,
)
from node.evalGraph.LanguageQualityEvalNode.language_quality_eval_node_config import (
    language_quality_eval_node,
)
from node.evalGraph.MergeScoresNode.merge_scores_node_config import merge_scores_node
from node.evalGraph.ValidityNode.validity_node_config import validity_node
from node.followUpDecisionGraph.followup_graph_helper import is_clarification_reason
from node.state_models import QuestionAssetContext, QuestionContext, SpeakingInput, TopicContext
from node.state_models.pronunciation import FormattedPronunciationResult
from infra.database import archive_store
from realtime.attempt_connection import get_attempt_connection
from schemas.enums import SpeakingMode
from utils.speech_client import (
    unwrap_language_tags,
)

logger = logging.getLogger(__name__)


def _real_transcript_for_turn(result: Dict[str, Any]) -> str:
    """The turn's actual transcribed text, from start_node's own Azure transcription.

    turn.transcript (the raw field on the Kafka request payload, coming from vox) is
    never populated -- vox's archive flow never transcribes anything itself, only
    Python's start_node does, per-turn, at graph.invoke() time. Anything that wants
    "what did the student actually say for this turn" must read it from here
    (result["speaking_input"].transcribed_text), not from turn.transcript.

    Delegates to display_transcript (exam_event_builder.py), which is just
    speaking_input.transcribed_text -- Azure's own transcription, no LLM rewrite step.

    Unwrapped for display: transcribed_text may still contain "[VI: ...]"
    code-switch markers (see speech_client._wrap_non_target_segment) that scoring
    nodes rely on, but a reader just wants to read the sentence naturally --
    unwrap_language_tags drops the "[XX: ]" bookkeeping punctuation while keeping the
    actual words in place.
    """
    return unwrap_language_tags(display_transcript(result, result.get("speaking_input")))


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


def _format_elapsed(total_seconds: float) -> str:
    total_seconds = max(0, int(round(total_seconds)))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def _build_dialogue_transcript(per_turn_results: List[Tuple[Any, Dict[str, Any]]]) -> str:
    """Timestamped AI/User dialogue transcript, e.g.:
        [00:00] AI: What is your favorite color?
        [00:05] User: I like blue because the sky is blue.

    Used only as additional context for the coherence judgment (to judge whether
    the answer coherently follows what was actually asked, including
    follow-ups) -- never fed into grammar/lexical/answer-length, which must
    only see the student's own words (see _combine_transcript above).
    Timestamps are approximate: each AI line inherits the running elapsed
    time, then the elapsed time advances by that turn's own duration_seconds
    after the User line -- ordering/pacing matters here, not precision.
    """
    lines: List[str] = []
    elapsed = 0.0
    for turn, result in sorted(per_turn_results, key=lambda item: item[0].turn_order):
        if turn.prompt_text and turn.prompt_text.strip():
            lines.append(f"[{_format_elapsed(elapsed)}] AI: {turn.prompt_text.strip()}")
        text = _real_transcript_for_turn(result).strip()
        if text:
            lines.append(f"[{_format_elapsed(elapsed)}] User: {text}")
        elapsed += float(turn.duration_seconds or 0)
    return "\n".join(lines)


def _total_duration_seconds(turns: List[Any]) -> int:
    return int(round(sum(float(turn.duration_seconds or 0) for turn in turns)))


def _aggregate_response_duration(turns: List[Any]) -> Optional[float]:
    durations = [
        float(turn.duration_seconds)
        for turn in turns
        if turn.duration_seconds is not None
    ]
    return sum(durations) if durations else None


def _severity_rank(value: Optional[str]) -> int:
    ranks = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    return ranks.get(str(value or "none").lower(), 0)


def _deduplicate_turn_rules(rules: List[RuleResult]) -> List[RuleResult]:
    grouped: Dict[str, List[RuleResult]] = {}
    for rule in rules:
        grouped.setdefault(rule.rule_id, []).append(rule)

    merged: List[RuleResult] = []
    for rule_id, matches in grouped.items():
        selected = max(matches, key=lambda item: _severity_rank(item.severity))
        turn_orders = sorted(
            {
                int(rule.evidence["turn_order"])
                for rule in matches
                if isinstance(rule.evidence.get("turn_order"), (int, float))
            }
        )
        evidence = {
            key: value
            for key, value in (selected.evidence or {}).items()
            if key not in {"turn_order", "turn_type"}
        }
        evidence.update(
            {
                "turn_orders": turn_orders,
                "occurrence_count": len(matches),
                "aggregation": "deduplicated_by_rule_id",
            }
        )
        merged.append(
            selected.model_copy(
                update={
                    "rule_id": rule_id,
                    "evidence": evidence,
                    "target_criteria": sorted(
                        {
                            criterion
                            for rule in matches
                            for criterion in rule.target_criteria
                        }
                    ),
                }
            )
        )
    return merged


def _merge_turn_validity(
    per_turn_results: List[Tuple[Any, Dict[str, Any]]],
    aggregate_validity: ValidityResult,
) -> ValidityResult:
    """Keep aggregate validity authoritative and preserve only turn-local safety flags."""
    rules = [
        rule.model_copy(
            update={
                "evidence": {
                    **(rule.evidence or {}),
                    "aggregation": "complete_answer",
                }
            }
        )
        for rule in aggregate_validity.rule_results
    ]
    for turn, result in sorted(per_turn_results, key=lambda item: item[0].turn_order):
        validity = result.get("validity")
        if not isinstance(validity, ValidityResult):
            continue
        for rule in validity.rule_results:
            if rule.rule_id != "safety.profanity_or_abuse":
                continue
            evidence = {
                **(rule.evidence or {}),
                "turn_order": turn.turn_order,
                "turn_type": turn.turn_type or ("MAIN" if turn.turn_order == 1 else "FOLLOWUP"),
                "aggregation": "turn_safety_backstop",
            }
            rules.append(rule.model_copy(update={"evidence": evidence}))

    merged_rules = _deduplicate_turn_rules(rules)
    has_blocking_reject = any(
        rule.blocking and rule.action == "reject_or_zero"
        for rule in merged_rules
    )
    valid_for_scoring = (
        bool(aggregate_validity.valid_for_scoring)
        and not has_blocking_reject
    )
    has_penalty = any(
        rule.action == "score_with_penalty"
        for rule in merged_rules
    )
    highest = max(
        (_severity_rank(rule.severity) for rule in merged_rules),
        default=_severity_rank(aggregate_validity.overall_severity),
    )
    severity_by_rank = ["none", "low", "medium", "high", "critical"]
    score_caps = dict(aggregate_validity.score_caps)
    if not valid_for_scoring:
        score_caps.update(
            {
                "pronunciation_max": 0,
                "fluency_max": 0,
                "grammar_max": 0,
                "vocabulary_max": 0,
                "coherence_max": 0,
            }
        )

    return ValidityResult(
        valid_for_scoring=valid_for_scoring,
        action=(
            "reject_or_zero"
            if not valid_for_scoring
            else "score_with_penalty"
            if has_penalty
            else "score"
        ),
        overall_severity=severity_by_rank[highest],
        rule_results=merged_rules,
        flags=aggregate_validity.flags,
        score_caps=score_caps,
        penalties=aggregate_validity.penalties,
        transcript_source="multi_turn_transcribed_text",
        transcript_word_count=aggregate_validity.transcript_word_count,
        notes=[
            *aggregate_validity.notes,
            "Multi-turn validity was evaluated once against the complete answer.",
        ],
    )


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
    realtime_transcript: Optional[str],
    realtime_transcript_confidence: Optional[float],
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
            realtime_transcript=realtime_transcript,
            realtime_transcript_confidence=realtime_transcript_confidence,
        ),
        "status": "idle",
        "metadata": {
            "request_payload": raw_payload,
            "turn_order": turn.turn_order,
            "actual_response_seconds": turn.duration_seconds,
            "validity_scope": (
                "turn_fragment"
                if len(request_event.payload.turns) > 1
                else "complete_answer"
            ),
        },
    }


async def _evaluate_turn(
    graph: Any,
    archive_graph: Any,
    request_event: ExamAttemptEvaluationRequestedEvent,
    turn: Any,
    conversation_transcript: str,
    raw_payload: Dict[str, Any],
) -> Dict[str, Any]:
    local_audio_path = await download_from_s3_async(turn.audio_ref)
    try:
        realtime_transcript, realtime_transcript_confidence = await archive_store.get_realtime_transcript(
            archive_graph, request_event.answer_id, turn.turn_order,
        )

        initial_state = _build_initial_state(
            request_event,
            turn,
            local_audio_path,
            conversation_transcript,
            raw_payload,
            realtime_transcript,
            realtime_transcript_confidence,
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
    total_response_seconds: Optional[float],
) -> Dict[str, Any]:
    """Grade grammar/lexical/coherence/answer-length ONCE, against the whole
    multi-turn answer -- not per turn.

    start_node (run inside each per-turn graph.invoke above) unconditionally
    re-transcribes speaking_input.transcribed_text from that turn's own
    audio_path, so there is no way to get a full-answer merged transcript
    through the compiled graph's normal entry point. Pronunciation is evaluated
    per audio turn; this function runs complete-answer validity, answer-length,
    and language quality against the merged student transcript.
    """
    base_speaking_input = per_turn_results[0][1]["speaking_input"]
    speaking_input = base_speaking_input.model_copy(update={
        "transcribed_text": merged_transcript,
        "conversation_transcript": dialogue_transcript,
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
        "metadata": (
            {
                "actual_response_seconds": total_response_seconds,
                "validity_scope": "complete_answer",
            }
            if total_response_seconds is not None
            else {"validity_scope": "complete_answer"}
        ),
        "validity": ValidityResult(),
    }

    state = validity_node(state)
    aggregate_validity = state.get("validity")
    if not isinstance(aggregate_validity, ValidityResult):
        raise RuntimeError("aggregate validity evaluation did not return ValidityResult")
    state["validity"] = _merge_turn_validity(
        per_turn_results,
        aggregate_validity,
    )

    # Invalid complete answers do not need answer-length or language-quality calls.
    if not state["validity"].valid_for_scoring:
        return {**state, **merge_scores_node(state)}

    delta = answer_length_analysis_node(state)
    if "metadata" in delta:
        delta = {
            **delta,
            "metadata": {
                **(state.get("metadata") or {}),
                **(delta.get("metadata") or {}),
            },
        }
    state = {**state, **delta}

    delta = language_quality_eval_node(state)
    if "metadata" in delta:
        delta = {
            **delta,
            "metadata": {
                **(state.get("metadata") or {}),
                **(delta.get("metadata") or {}),
            },
        }
    state = {**state, **delta}

    state = {**state, **merge_scores_node(state)}
    if state.get("status") == "error":
        raise RuntimeError(state.get("error") or "aggregate text evaluation failed")

    return state


def _merge_multi_turn_criteria(
    aggregate_result: Dict[str, Any],
    per_turn_results: List[Tuple[Any, Dict[str, Any]]],
) -> Dict[str, Any]:
    criteria = build_criteria_payload(aggregate_result)
    if not criteria:
        return criteria

    validity = aggregate_result.get("validity")
    if isinstance(validity, ValidityResult) and (
        not validity.valid_for_scoring
        or validity.overall_severity == "critical"
    ):
        return {
            name: criterion.model_copy(update={
                "score": 0,
                "level": "not_scored",
                "status": "zeroed",
                "source": "rule",
                "note": "Điểm được đưa về 0 do câu trả lời vi phạm validity ở mức critical.",
            })
            for name, criterion in criteria.items()
        }

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
    turns: List[Any],
    per_turn_results: List[Tuple[Any, Dict[str, Any]]],
    total_duration_seconds: int,
) -> ExamAttemptEvaluationCompletedEvent:
    speaking_input = aggregate_result.get("speaking_input")
    feedback_summary, suggestions = build_feedback_payload(
        aggregate_result,
        speaking_input,
        audio_path=aggregate_audio_path,
    )
    results_by_turn_order = {turn.turn_order: result for turn, result in per_turn_results}
    completed_turns = []
    for turn in turns:
        result = results_by_turn_order.get(turn.turn_order, {})
        transcript = (
            _real_transcript_for_turn(result)
            if result
            else (turn.transcript or "")
        )
        completed_turns.append(
            build_turn_detail(
                result,
                turn_order=turn.turn_order,
                turn_type=turn.turn_type or "MAIN",
                prompt_text=turn.prompt_text,
                audio_url=turn.audio_ref,
                transcript=transcript,
                duration_seconds=turn.duration_seconds,
            )
        )

    return ExamAttemptEvaluationCompletedEvent(
        exam_attempt_id=request_event.exam_attempt_id,
        answer_id=request_event.answer_id,
        question_id=request_event.question_id,
        payload=ExamAttemptEvaluationCompletedPayload(
            turns=completed_turns,
            criteria=_merge_multi_turn_criteria(aggregate_result, per_turn_results),
            signals=build_signals(
                aggregate_result,
                speaking_input,
                duration_seconds=total_duration_seconds,
                turn_results=[result for _turn, result in per_turn_results],
            ),
            validity=aggregate_result.get("validity"),
            feedback_summary=feedback_summary,
            suggestions=suggestions,
            model_version="gpt-5.4+claude-sonnet-4-6",
            prompt_version="language-quality-v2",
            evaluated_at=build_completed_event(
                aggregate_result,
                speaking_input,
                audio_path=aggregate_audio_path,
            ).payload.evaluated_at,
        ),
    )


def _build_no_answer_completed_event(
    request_event: ExamAttemptEvaluationRequestedEvent,
) -> ExamAttemptEvaluationCompletedEvent:
    """A question with zero archived turns (request_payload.turns is empty) means the student
    never answered it at all -- e.g. the exam session ended/disconnected before they got to speak
    (task/exam-interrupted-session-grading.txt). That is a normal, expected outcome that must
    still produce a graded result, not an evaluation failure: previously the bare `if not turns`
    check below raised ValueError, which retried KAFKA_MAX_RETRY times against the same
    never-changing payload (always failing identically) and ended in
    ExamAttemptEvaluationFailedEvent -- an "evaluation broke" state, not "student skipped this,
    scored 0" like it should be.

    Mirrors ValidityNode's existing per-turn "audio.no_speech" rule (action=reject_or_zero,
    status=zeroed) but at the whole-question level, since there is no turn/audio at all here to
    run that per-turn rule against."""
    no_turns_rule = RuleResult(
        rule_id="answer.no_turns",
        category="validity",
        status="triggered",
        severity="none",
        action="reject_or_zero",
        message="Student did not answer this question (no turns recorded).",
    )
    validity = ValidityResult(
        valid_for_scoring=False,
        action="reject_or_zero",
        rule_results=[no_turns_rule],
        transcript_source="none",
        transcript_word_count=0,
    )
    signals = EvaluationSignals(duration_seconds=0, word_count=0, sentence_count=0, expected_min_words=0)
    criteria = {
        name: CriterionScore(score=0, level="not_scored", status="zeroed", source="rule")
        for name in ("pronunciation", "fluency", "grammar", "vocabulary", "coherence")
    }

    return ExamAttemptEvaluationCompletedEvent(
        exam_attempt_id=request_event.exam_attempt_id,
        answer_id=request_event.answer_id,
        question_id=request_event.question_id,
        payload=ExamAttemptEvaluationCompletedPayload(
            turns=[],
            criteria=criteria,
            signals=signals,
            validity=validity,
            feedback_summary="Hoc sinh khong tra loi cau hoi nay.",
            suggestions=[],
            model_version="rule-based-no-answer",
            prompt_version="v1",
            evaluated_at=datetime.now(timezone.utc).isoformat(),
        ),
    )


def _build_clarification_only_completed_event(
    request_event: ExamAttemptEvaluationRequestedEvent,
    turns: List[Any],
) -> ExamAttemptEvaluationCompletedEvent:
    clarification_rule = RuleResult(
        rule_id="answer.clarification_only",
        category="validity",
        status="triggered",
        severity="none",
        action="reject_or_zero",
        message="Only clarification-only turns were recorded for this question.",
    )
    validity = ValidityResult(
        valid_for_scoring=False,
        action="reject_or_zero",
        rule_results=[clarification_rule],
        transcript_source="archive",
        transcript_word_count=0,
    )
    criteria = {
        name: CriterionScore(score=0, level="not_scored", status="zeroed", source="rule")
        for name in ("pronunciation", "fluency", "grammar", "vocabulary", "coherence")
    }
    completed_turns = [
        build_turn_detail(
            {},
            turn_order=turn.turn_order,
            turn_type=turn.turn_type or "MAIN",
            prompt_text=turn.prompt_text,
            audio_url=turn.audio_ref,
            transcript=turn.transcript or "",
            duration_seconds=turn.duration_seconds,
        )
        for turn in turns
    ]

    return ExamAttemptEvaluationCompletedEvent(
        exam_attempt_id=request_event.exam_attempt_id,
        answer_id=request_event.answer_id,
        question_id=request_event.question_id,
        payload=ExamAttemptEvaluationCompletedPayload(
            turns=completed_turns,
            criteria=criteria,
            signals=EvaluationSignals(
                duration_seconds=_total_duration_seconds(turns),
                word_count=0,
                sentence_count=0,
                expected_min_words=0,
            ),
            validity=validity,
            feedback_summary="The student only used clarification or repair turns for this question.",
            suggestions=[],
            model_version="rule-based-clarification-only",
            prompt_version="v1",
            evaluated_at=datetime.now(timezone.utc).isoformat(),
        ),
    )


async def start_exam_attempt_consumer(app, instance_label: str = "0"):
    consumer = await get_topic_consumer_instance(
        settings.KAFKA_EXAM_REQUEST_TOPIC,
        group_id=settings.KAFKA_EXAM_CONSUMER_GROUP,
        instance_label=instance_label,
    )
    graph = app.state.graph
    archive_graph = app.state.archive_graph

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
                    # Not an error -- the student simply never answered this question (see
                    # _build_no_answer_completed_event's docstring). Publish a 0-point completed
                    # result directly instead of raising into the retry/failed path below, which
                    # would just fail identically every retry since this payload never changes.
                    logger.info(
                        "[exam-consumer] no turns recorded, publishing 0-point result answer_id=%s exam_attempt_id=%s",
                        request_event.answer_id, request_event.exam_attempt_id,
                    )
                    await publish_exam_attempt_evaluation_completed(
                        _build_no_answer_completed_event(request_event)
                    )
                    await consumer.commit()
                    break

                logger.info(
                    "[exam-consumer] received answer_id=%s exam_attempt_id=%s turns=%d",
                    request_event.answer_id, request_event.exam_attempt_id, len(turns),
                )

                aggregate_audio_path = turns[0].audio_ref
                scoreable_turns = [
                    turn for turn in turns
                    if not is_clarification_reason(getattr(turn, "decision_reason", None))
                ]

                async def _evaluate_one(turn):
                    logger.info(
                        "[exam-consumer] evaluating turn %d answer_id=%s",
                        turn.turn_order,
                        request_event.answer_id,
                    )
                    # Multi-turn fragments only run transcription, deterministic local
                    # validity gates, and pronunciation. Complete-answer semantic validity,
                    # answer length, and language quality run once after all real per-turn
                    # transcripts are available.
                    result = await _evaluate_turn(
                        graph,
                        archive_graph,
                        request_event,
                        turn,
                        "",
                        payload,
                    )
                    logger.info(
                        "[exam-consumer] turn %d done answer_id=%s",
                        turn.turn_order,
                        request_event.answer_id,
                    )
                    return turn, result

                per_turn_results: List[Tuple[Any, Dict[str, Any]]] = list(
                    await asyncio.gather(
                        *[_evaluate_one(turn) for turn in scoreable_turns]
                    )
                )

                if not scoreable_turns:
                    logger.info(
                        "[exam-consumer] clarification-only answer, publishing zero-point result answer_id=%s",
                        request_event.answer_id,
                    )
                    await publish_exam_attempt_evaluation_completed(
                        _build_clarification_only_completed_event(request_event, turns)
                    )
                    await consumer.commit()
                    break

                merged_transcript = _combine_transcript(per_turn_results)
                dialogue_transcript = _build_dialogue_transcript(per_turn_results)

                logger.info("[exam-consumer] running aggregate text evaluation answer_id=%s", request_event.answer_id)
                aggregate_result = await asyncio.to_thread(
                    _run_aggregate_text_evaluation,
                    per_turn_results,
                    merged_transcript,
                    dialogue_transcript,
                    _aggregate_response_duration(
                        [turn for turn, _result in per_turn_results]
                    ),
                )
                completed_event = _build_multi_turn_completed_event(
                    request_event,
                    aggregate_result,
                    aggregate_audio_path,
                    turns,
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


async def start_exam_attempt_force_end_consumer(app):
    consumer = await get_topic_consumer(
        settings.KAFKA_EXAM_FORCE_END_TOPIC,
        group_id=settings.KAFKA_EXAM_FORCE_END_CONSUMER_GROUP,
    )

    async for message in consumer:
        try:
            payload = json.loads(message.value.decode())
            event = ExamAttemptForceEndRequestedEvent.model_validate(payload)
            connection = get_attempt_connection(event.exam_attempt_id)
            if connection is None:
                logger.info(
                    "[exam-force-end-consumer] no local realtime connection exam_attempt_id=%s",
                    event.exam_attempt_id,
                )
            else:
                await connection.force_end(event.payload.reason or "")
                logger.info(
                    "[exam-force-end-consumer] forwarded force_end exam_attempt_id=%s",
                    event.exam_attempt_id,
                )
        except Exception:
            logger.exception(
                "[exam-force-end-consumer] failed topic=%s partition=%s offset=%s",
                message.topic,
                message.partition,
                message.offset,
            )
        finally:
            await consumer.commit()
