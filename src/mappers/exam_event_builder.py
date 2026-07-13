import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from events.exam_attempt_evaluation_completed import (
    ExamAttemptEvaluationCompletedEvent,
    ExamAttemptEvaluationCompletedPayload,
)
from events.exam_attempt_evaluation_shared import (
    EvaluationSignals,
    PronunciationOverallScores,
    TurnDetail,
)
from mappers.assessment_response_adapter import adapt_current_response_to_ui_response
from node.state_models.pronunciation import PhonemeFeedback, WordFeedback
from node.state_models.speaking_input import SpeakingInput
from schemas.scoring import CriterionScore
from schemas.validity import ValidityResult


def _to_word_feedback(word: Dict[str, Any]) -> WordFeedback:
    phonemes = [
        PhonemeFeedback(
            phoneme=p.get("phoneme", ""),
            accuracy_score=p.get("score"),
            color=p.get("color"),
            level=p.get("level"),
            note=p.get("note"),
        )
        for p in word.get("phonemes", [])
    ]
    return WordFeedback(
        word=word.get("word", ""),
        accuracy_score=word.get("accuracy_score"),
        effective_score=word.get("effective_score"),
        error_type=word.get("error_type"),
        color=word.get("color"),
        level=word.get("level"),
        error_note=word.get("error_note"),
        has_critical_issue=word.get("has_critical_issue"),
        phonemes=phonemes,
    )


def _word_count(text: Optional[str]) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def _extract_pronunciation_parts(
    result: Dict[str, Any],
) -> Tuple[Dict[str, CriterionScore], Dict[str, Optional[float]], List[WordFeedback]]:
    pronunciation_result = result.get("pronunciation_result")
    if pronunciation_result is None:
        return {}, {}, []

    criteria = {
        "pronunciation": pronunciation_result.criteria.pronunciation,
        "fluency": pronunciation_result.criteria.fluency,
        "grammar": pronunciation_result.criteria.grammar,
        "vocabulary": pronunciation_result.criteria.vocabulary,
        "coherence": pronunciation_result.criteria.coherence,
    }
    overall = pronunciation_result.overall or {}
    word_feedback = [_to_word_feedback(w) for w in pronunciation_result.word_feedback or []]
    return criteria, overall, word_feedback


def _build_old_response(
    result: Dict[str, Any],
    speaking_input: SpeakingInput,
    *,
    audio_path: str,
) -> Dict[str, Any]:
    question = speaking_input.question
    topic = speaking_input.topic
    return {
        "status": result.get("status"),
        "error": result.get("error"),
        "audio_path": audio_path,
        "mode": speaking_input.mode,
        "reference_text": speaking_input.reference_text,
        "question": {
            "question_id": speaking_input.question_id,
            "question_text": question.question_text if question else None,
            "question_type": question.question_type if question else None,
            "difficulty_level": question.difficulty_level if question else None,
            "duration_seconds": question.duration_seconds if question else None,
            "min_response_seconds": question.min_response_seconds if question else None,
            "max_response_seconds": question.max_response_seconds if question else None,
        },
        "topic": {
            "topic_id": topic.topic_id if topic else None,
            "topic_name": topic.topic_name if topic else None,
            "topic_description": topic.topic_description if topic else None,
        },
        "result": result.get("pronunciation_result"),
        "metadata": result.get("metadata", {}),
        "validity": result.get("validity"),
    }


def build_feedback_payload(
    result: Dict[str, Any],
    speaking_input: Optional[SpeakingInput],
    *,
    audio_path: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    if speaking_input is None:
        return "", []

    ui_response = adapt_current_response_to_ui_response(
        _build_old_response(result, speaking_input, audio_path=audio_path)
    )
    feedback = ui_response.get("feedback") or {}
    return feedback.get("summary") or "", feedback.get("suggestions") or []


def build_criteria_payload(result: Dict[str, Any]) -> Dict[str, CriterionScore]:
    criteria, _, _ = _extract_pronunciation_parts(result)
    return criteria


def _clamp_unit(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def _average(values: List[Optional[float]]) -> Optional[float]:
    normalized = [value for value in values if value is not None]
    if not normalized:
        return None
    return sum(normalized) / len(normalized)


def _compute_ai_confidence(result: Dict[str, Any]) -> Optional[float]:
    metadata = result.get("metadata") or {}
    return _average(
        [
            _clamp_unit(metadata.get("coherence_confidence")),
            _clamp_unit(metadata.get("grammar_confidence")),
            _clamp_unit(metadata.get("vocabulary_confidence")),
        ]
    )


def _compute_audio_quality(metrics: Dict[str, Any]) -> Optional[float]:
    asr_confidence = _clamp_unit(metrics.get("asr_confidence_avg"))
    silence_ratio = _clamp_unit(metrics.get("silence_ratio"))

    if asr_confidence is not None and silence_ratio is not None:
        return 0.7 * asr_confidence + 0.3 * (1.0 - silence_ratio)
    if asr_confidence is not None:
        return asr_confidence
    if silence_ratio is not None:
        return 1.0 - silence_ratio
    return None


def build_signals(
    result: Dict[str, Any],
    speaking_input: Optional[SpeakingInput] = None,
    *,
    duration_seconds: int = 0,
) -> EvaluationSignals:
    if speaking_input is None:
        speaking_input = result.get("speaking_input")

    metrics = speaking_input.answer_length_metrics if speaking_input is not None else {}
    metrics = metrics or {}
    return EvaluationSignals(
        duration_seconds=duration_seconds,
        word_count=metrics.get("word_count", 0),
        sentence_count=metrics.get("sentence_count", 0),
        length_ratio=metrics.get("length_ratio"),
        expected_min_words=metrics.get("expected_min_words", 0),
        asr_confidence_avg=metrics.get("asr_confidence_avg"),
        topic_relevance_score=metrics.get("topic_relevance_score"),
        off_topic_ratio=metrics.get("off_topic_ratio"),
        code_switching_ratio=metrics.get("code_switching_ratio"),
        speech_rate=metrics.get("speech_rate"),
        ai_confidence=_compute_ai_confidence(result),
        audio_quality=_compute_audio_quality(metrics),
        silence_ratio=metrics.get("silence_ratio"),
    )


def display_transcript(result: Dict[str, Any], speaking_input: Optional[SpeakingInput]) -> str:
    """The transcript to show/score against for a turn: Azure's own transcribed_text
    (speech_client.transcribe(), already auto-detect + language-tagged for code-switched
    Vietnamese) -- no LLM rewrite step in between. `result` is unused now that there's no
    violation-dependent choice to make, kept in the signature so callers don't need updating."""
    if speaking_input is None:
        return ""
    return speaking_input.transcribed_text or ""


def build_turn_detail(
    result: Dict[str, Any],
    *,
    turn_order: int,
    turn_type: str,
    prompt_text: Optional[str],
    audio_url: str,
    transcript: Optional[str],
    duration_seconds: Optional[int] = None,
    turn_id: Optional[str] = None,
) -> TurnDetail:
    _, overall, word_feedback = _extract_pronunciation_parts(result)
    transcript_value = transcript or ""
    return TurnDetail(
        turn_id=turn_id,
        turn_order=turn_order,
        turn_type=turn_type,
        prompt_text=prompt_text,
        audio_url=audio_url,
        transcript=transcript_value,
        word_count=_word_count(transcript_value),
        duration_seconds=duration_seconds,
        pronunciation_overall=PronunciationOverallScores(
            accuracy_score=overall.get("accuracy_score"),
            fluency_score=overall.get("fluency_score"),
            prosody_score=overall.get("prosody_score"),
            pron_score=overall.get("pron_score"),
            completeness_score=overall.get("completeness_score"),
        ),
        word_feedback=word_feedback,
    )


def build_completed_event(
    result: Dict[str, Any],
    speaking_input: Optional[SpeakingInput],
    *,
    audio_path: str,
) -> Optional[ExamAttemptEvaluationCompletedEvent]:
    if speaking_input is None:
        return None

    criteria = build_criteria_payload(result)
    signals = build_signals(result, speaking_input)
    feedback_summary, suggestions = build_feedback_payload(
        result,
        speaking_input,
        audio_path=audio_path,
    )
    turn = build_turn_detail(
        result,
        turn_order=1,
        turn_type="MAIN",
        prompt_text=speaking_input.question.question_text if speaking_input.question else None,
        audio_url=audio_path,
        transcript=display_transcript(result, speaking_input),
    )

    return ExamAttemptEvaluationCompletedEvent(
        exam_attempt_id=speaking_input.exam_attempt_id or "test-exam-attempt",
        answer_id=speaking_input.answer_id or "test-answer",
        question_id=speaking_input.question_id or "test-question",
        payload=ExamAttemptEvaluationCompletedPayload(
            turns=[turn],
            criteria=criteria,
            signals=signals,
            validity=result.get("validity") or ValidityResult(),
            feedback_summary=feedback_summary,
            suggestions=suggestions,
            model_version="gpt-4o",
            prompt_version="v1",
            evaluated_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
