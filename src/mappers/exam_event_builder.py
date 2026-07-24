import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from events.exam_attempt_evaluation_completed import (
    ExamAttemptEvaluationCompletedEvent,
    ExamAttemptEvaluationCompletedPayload,
)
from events.exam_attempt_evaluation_shared import (
    ConfidenceCaseSignals,
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
        # answer_length_metrics is its own top-level GraphState key now (not nested inside
        # metadata -- see GraphState.py), added back in here so
        # assessment_response_adapter.py's existing metadata["answer_length_metrics"] lookup
        # keeps working unchanged.
        "metadata": {**result.get("metadata", {}), "answer_length_metrics": result.get("answer_length_metrics")},
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


def _compute_audio_quality(metrics: Dict[str, Any]) -> Optional[float]:
    # Audio soft scalar chỉ dùng tín hiệu âm học. ASR confidence/cross-ASR là triage
    # transcription, không phải chất lượng audio. Dùng worst-link thay vì average để
    # không pha loãng một thành phần âm thanh rất xấu.
    q_snr = _clamp_unit(metrics.get("q_snr"))
    q_speech = _clamp_unit(metrics.get("q_speech"))
    quality_parts = [value for value in (q_snr, q_speech) if value is not None]
    if quality_parts:
        return min(quality_parts)
    silence_ratio = _clamp_unit(metrics.get("silence_ratio"))
    if silence_ratio is not None:
        return 1.0 - silence_ratio
    return None


def _min_available(*values: Optional[float]) -> Optional[float]:
    available = [float(value) for value in values if value is not None]
    return min(available) if available else None


def _build_turn_confidence_case(result: Dict[str, Any]) -> ConfidenceCaseSignals:
    metadata = result.get("metadata") or {}
    metrics = result.get("answer_length_metrics") or {}
    speaking_input = result.get("speaking_input")

    has_realtime = bool(
        speaking_input is not None and speaking_input.realtime_transcript
    )
    realtime_confidence = (
        _clamp_unit(speaking_input.realtime_transcript_confidence)
        if has_realtime
        else None
    )
    # Audio gate độc lập với nhánh ASR, nên luôn truyền q_snr/q_speech khi đo được.
    q_snr = _clamp_unit(metrics.get("q_snr"))
    q_speech = _clamp_unit(metrics.get("q_speech"))
    c_ref = _clamp_unit(metadata.get("c_ref"))
    c_align = _clamp_unit(metadata.get("c_align"))
    c_align_accuracy = _clamp_unit(metadata.get("c_align_accuracy"))
    c_align_coverage = _clamp_unit(metadata.get("c_align_coverage"))
    c_align_timing = _clamp_unit(metadata.get("c_align_timing"))

    # c_pf_branch = min(c_ref, c_align), theo ĐÚNG giả mã cuối của research
    # (02-ket-qua.md, requires_human_review): KHÔNG ép về 0 khi ASR thấp. Việc "ASR hỏng ->
    # chuyển người" đã được ConfidenceReviewCalculator xử lý ĐỘC LẬP qua ngưỡng asrNolog/cAsrLog
    # riêng (nó không hề đọc c_pf_branch). Bản cũ ép c_pf_branch=0 ngay ở ngưỡng SOFT 0.80 (nhánh
    # nolog) -- vừa sai ngưỡng (hard đúng ra là 0.60), vừa sai cơ chế -- khiến overallConfidence
    # (min mọi c-value phía Java) LUÔN = 0 dù cRef/cAlign rất tốt.
    pf_components = [value for value in (c_ref, c_align) if value is not None]
    c_pf_branch = min(pf_components) if pf_components else None

    return ConfidenceCaseSignals(
        c_asr_log=realtime_confidence,
        q_snr=q_snr,
        q_speech=q_speech,
        clipping_ratio=_clamp_unit(metrics.get("clipping_ratio")),
        c_ref=c_ref,
        c_align=c_align,
        c_align_accuracy=c_align_accuracy,
        c_align_coverage=c_align_coverage,
        c_align_timing=c_align_timing,
        c_pf_branch=c_pf_branch,
    )


def _aggregate_confidence_cases(
    result: Dict[str, Any],
    turn_results: Optional[List[Dict[str, Any]]],
) -> Optional[ConfidenceCaseSignals]:
    cases = [
        _build_turn_confidence_case(item)
        for item in (turn_results or [result])
    ]
    metadata = result.get("metadata") or {}

    def minimum(field: str) -> Optional[float]:
        return _min_available(*(getattr(case, field) for case in cases))

    clipping_values = [
        case.clipping_ratio for case in cases if case.clipping_ratio is not None
    ]
    confidence_case = ConfidenceCaseSignals(
        c_asr_log=minimum("c_asr_log"),
        q_snr=minimum("q_snr"),
        q_speech=minimum("q_speech"),
        clipping_ratio=max(clipping_values) if clipping_values else None,
        c_ref=minimum("c_ref"),
        c_align=minimum("c_align"),
        c_align_accuracy=minimum("c_align_accuracy"),
        c_align_coverage=minimum("c_align_coverage"),
        c_align_timing=minimum("c_align_timing"),
        c_pf_branch=minimum("c_pf_branch"),
        c_grammar=_clamp_unit(metadata.get("grammar_confidence")),
        c_vocabulary=_clamp_unit(metadata.get("vocabulary_confidence")),
        c_discourse=_clamp_unit(metadata.get("coherence_confidence")),
        grammar_score_delta=metadata.get("grammar_score_delta"),
        vocabulary_score_delta=metadata.get("lexical_score_delta"),
        discourse_score_delta=metadata.get("coherence_score_delta"),
    )
    return (
        confidence_case
        if any(value is not None for value in confidence_case.model_dump().values())
        else None
    )


def build_signals(
    result: Dict[str, Any],
    speaking_input: Optional[SpeakingInput] = None,
    *,
    duration_seconds: int = 0,
    turn_results: Optional[List[Dict[str, Any]]] = None,
) -> EvaluationSignals:
    if speaking_input is None:
        speaking_input = result.get("speaking_input")

    # answer_length_metrics is its own top-level GraphState key now, written by
    # answer_length_analysis_node directly (not nested inside speaking_input -- that node
    # runs in parallel with pronunciation_eval, see graphConfig.build_graph, and no longer
    # mutates the shared speaking_input object).
    metrics = result.get("answer_length_metrics") or {}
    evidence_reason_codes: List[str] = []
    if metrics.get("length_category") == "too_short":
        evidence_reason_codes.append("ANSWER_TOO_SHORT")
    actual_response_seconds = metrics.get("actual_response_seconds")
    min_response_seconds = metrics.get("min_response_seconds")
    length_ratio = metrics.get("length_ratio")
    try:
        duration_ratio = (
            float(actual_response_seconds) / float(min_response_seconds)
            if actual_response_seconds is not None and min_response_seconds
            else None
        )
    except (TypeError, ValueError, ZeroDivisionError):
        duration_ratio = None
    if duration_ratio is not None and duration_ratio < 0.5 and (length_ratio is None or length_ratio < 0.9):
        evidence_reason_codes.append("RESPONSE_DURATION_TOO_SHORT")
    code_switching_ratio = _clamp_unit(metrics.get("code_switching_ratio"))
    if code_switching_ratio is not None and code_switching_ratio > 0.50:
        evidence_reason_codes.append("TARGET_LANGUAGE_EVIDENCE_TOO_LOW")
    silence_ratio = _clamp_unit(metrics.get("silence_ratio"))
    if silence_ratio is not None and silence_ratio >= 0.70:
        evidence_reason_codes.append("SPEECH_EVIDENCE_TOO_LOW")

    return EvaluationSignals(
        duration_seconds=duration_seconds,
        word_count=metrics.get("word_count", 0),
        sentence_count=metrics.get("sentence_count", 0),
        length_ratio=metrics.get("length_ratio"),
        expected_min_words=metrics.get("expected_min_words", 0),
        asr_confidence_avg=metrics.get("asr_confidence_avg"),
        topic_relevance_score=metrics.get("topic_relevance_score"),
        off_topic_ratio=metrics.get("off_topic_ratio"),
        code_switching_ratio=code_switching_ratio,
        speech_rate=metrics.get("speech_rate"),
        audio_quality=_compute_audio_quality(metrics),
        silence_ratio=silence_ratio,
        evidence_status=(
            "INSUFFICIENT_EVIDENCE" if evidence_reason_codes else "SUFFICIENT"
        ),
        evidence_reason_codes=evidence_reason_codes,
        confidence_case=_aggregate_confidence_cases(result, turn_results),
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
    duration_seconds: Optional[float] = None,
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
            model_version="gpt-4o+claude-sonnet-4-6",
            prompt_version="language-quality-v2",
            evaluated_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
