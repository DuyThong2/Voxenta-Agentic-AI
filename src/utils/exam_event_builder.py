from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from infra.message_broker.events.completed import (
    ExamAttemptEvaluationCompletedEvent,
    ExamAttemptEvaluationCompletedPayload,
)
from infra.message_broker.events.shared import (
    EvaluationSignals,
    PronunciationOverallScores,
    TurnDetail,
)
from node.state_models.pronunciation import PhonemeFeedback, WordFeedback
from node.state_models.speaking_input import SpeakingInput
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


def build_completed_event(
    result: Dict[str, Any],
    speaking_input: Optional[SpeakingInput],
    *,
    audio_path: str,
) -> Optional[ExamAttemptEvaluationCompletedEvent]:
    """Build the Kafka-shaped completed event from a single-turn graph result.

    Used by the HTTP test endpoints so the event shape can be inspected without
    a real Kafka/multi-turn pipeline — `turns` always has exactly 1 MAIN turn here.
    """
    if speaking_input is None:
        return None

    pronunciation_result = result.get("pronunciation_result")
    validity = result.get("validity") or ValidityResult()
    metrics = speaking_input.answer_length_metrics or {}

    criteria: Dict[str, Any] = {}
    overall: Dict[str, Optional[float]] = {}
    word_feedback: List[WordFeedback] = []
    if pronunciation_result is not None:
        criteria = {
            "pronunciation": pronunciation_result.criteria.pronunciation,
            "fluency": pronunciation_result.criteria.fluency,
            "grammar": pronunciation_result.criteria.grammar,
            "vocabulary": pronunciation_result.criteria.vocabulary,
            "coherence": pronunciation_result.criteria.coherence,
        }
        overall = pronunciation_result.overall or {}
        word_feedback = [_to_word_feedback(w) for w in pronunciation_result.word_feedback or []]

    turn = TurnDetail(
        turn_order=1,
        turn_type="MAIN",
        prompt_text=speaking_input.question.question_text if speaking_input.question else None,
        audio_url=audio_path,
        transcript=speaking_input.transcribed_text or speaking_input.corrected_transcript or "",
        word_count=metrics.get("word_count", 0),
        pronunciation_overall=PronunciationOverallScores(
            accuracy_score=overall.get("accuracy_score"),
            fluency_score=overall.get("fluency_score"),
            prosody_score=overall.get("prosody_score"),
            pron_score=overall.get("pronunciation_score"),
            completeness_score=overall.get("completeness_score"),
        ),
        word_feedback=word_feedback,
    )

    signals = EvaluationSignals(
        word_count=metrics.get("word_count", 0),
        sentence_count=metrics.get("sentence_count", 0),
        length_ratio=metrics.get("length_ratio"),
        expected_min_words=metrics.get("expected_min_words", 0),
    )

    return ExamAttemptEvaluationCompletedEvent(
        exam_attempt_id=speaking_input.exam_attempt_id or "test-exam-attempt",
        answer_id=speaking_input.answer_id or "test-answer",
        question_id=speaking_input.question_id or "test-question",
        payload=ExamAttemptEvaluationCompletedPayload(
            turns=[turn],
            criteria=criteria,
            signals=signals,
            validity=validity,
            model_version="gpt-4o",
            prompt_version="v1",
            evaluated_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
