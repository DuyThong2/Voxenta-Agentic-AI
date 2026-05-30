"""Adapter to convert current graph responses into the standardized UI API contract."""

from typing import Any, Dict, List, Optional

from schemas.scoring import CriteriaScores
from schemas.ui_response import UIResponse
from schemas.validity import ValidityResult
from utils.schema_mapper import (
    build_validity_result_from_metrics,
    build_validity_result_from_payload,
    normalize_rule_result,
    safe_score,
    safe_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def level_from_score(score: Optional[float]) -> str:
    if score is None:
        return "not_scored"
    if score >= 85:
        return "excellent"
    if score >= 80:
        return "good"
    if score >= 60:
        return "fair"
    if score >= 40:
        return "limited"
    return "needs_improvement"


def format_phoneme_list(phonemes: List[str]) -> str:
    """Format a list of phoneme symbols into a human-readable string."""
    sounds = [f"/{p}/" for p in phonemes if p]
    if not sounds:
        return ""
    if len(sounds) == 1:
        return sounds[0]
    if len(sounds) == 2:
        return f"{sounds[0]} and {sounds[1]}"
    return f"{', '.join(sounds[:-1])}, and {sounds[-1]}"


# ---------------------------------------------------------------------------
# Criterion normalization
# ---------------------------------------------------------------------------

def normalize_criterion(item: Any, default_source: str = "unknown") -> Dict[str, Any]:
    if not isinstance(item, dict):
        item = {}

    score = safe_score(item.get("score"))
    status_value = item.get("status")
    if status_value not in {"scored", "not_scored", "zeroed", "capped"}:
        status_value = "scored" if score is not None else "not_scored"

    source_value = item.get("source")
    if source_value not in {"azure", "llm", "rule", "system"}:
        source_value = default_source if default_source in {"azure", "llm", "rule", "system"} else "system"

    result = {
        "score": score,
        "level": level_from_score(score),
        "status": status_value,
        "source": source_value,
        "subscores": item.get("subscores") or {},
        "note": item.get("note") or "",
        "suggestion": item.get("suggestion") or "",
    }
    return result


def build_default_criteria(result: Dict[str, Any]) -> Dict[str, Any]:
    """Build criteria from pronunciation_result.criteria (schema-aligned keys)."""
    raw_criteria = result.get("criteria") or {}

    criteria = {
        "pronunciation": normalize_criterion(raw_criteria.get("pronunciation") or {}, default_source="azure"),
        "fluency": normalize_criterion(raw_criteria.get("fluency") or {}, default_source="azure"),
        "grammar": normalize_criterion(raw_criteria.get("grammar") or {}, default_source="llm"),
        "vocabulary": normalize_criterion(raw_criteria.get("vocabulary") or {}, default_source="llm"),
        "coherence": normalize_criterion(raw_criteria.get("coherence") or {}, default_source="llm"),
    }

    return CriteriaScores.parse_obj(criteria).model_dump()


# ---------------------------------------------------------------------------
# Phoneme & word formatting
# ---------------------------------------------------------------------------

def format_phoneme(phoneme: Dict[str, Any], index: int) -> Dict[str, Any]:
    score = safe_score(phoneme.get("accuracy_score") or phoneme.get("score"))
    phoneme_symbol = safe_text(phoneme.get("phoneme")) or ""
    note = ""
    suggestion = ""
    if score is None:
        note = "The sound could not be scored."
        suggestion = "Listen to the target sound and try again."
    elif score < 60:
        note = f"The sound /{phoneme_symbol}/ needs clear improvement."
        suggestion = "Practice this sound with focused drills."
    elif score < 80:
        note = f"The sound /{phoneme_symbol}/ is understandable but should be practiced more."
        suggestion = "Repeat this sound with moderate emphasis."
    else:
        note = f"The sound /{phoneme_symbol}/ is good."
        suggestion = "Continue using this sound correctly."

    return {
        "index": index,
        "phoneme": phoneme_symbol,
        "score": score,
        "color": phoneme.get("color") or ("red" if score is not None and score < 60 else "yellow" if score is not None and score < 80 else "green" if score is not None else "gray"),
        "level": level_from_score(score),
        "note": note,
        "suggestion": suggestion,
    }


def format_word_feedback(word_obj: Dict[str, Any], index: int) -> Dict[str, Any]:
    # raw_azure_score: fallback chain for backward compat
    raw_azure_score = safe_score(
        word_obj.get("raw_azure_score")
        or word_obj.get("azure_score")
        or word_obj.get("accuracy_score")
    )

    phoneme_items: List[Dict[str, Any]] = []
    for idx, phoneme_obj in enumerate(word_obj.get("phonemes") or [], start=0):
        if isinstance(phoneme_obj, dict):
            phoneme_items.append(format_phoneme(phoneme_obj, idx))

    effective_score = raw_azure_score
    if word_obj.get("error_type") in ["Omission", "Insertion"]:
        effective_score = 0.0
    elif raw_azure_score is None and phoneme_items:
        scores = [p["score"] for p in phoneme_items if p["score"] is not None]
        effective_score = min(scores) if scores else None
    elif raw_azure_score is not None and phoneme_items:
        phoneme_scores = [p["score"] for p in phoneme_items if p["score"] is not None]
        if phoneme_scores:
            effective_score = min(raw_azure_score, min(phoneme_scores))

    effective_score = safe_score(effective_score)
    weak_phonemes = [p for p in phoneme_items if p["score"] is not None and p["score"] < 80]
    has_critical_issue = (
        word_obj.get("error_type") in ["Omission", "Insertion", "Mispronunciation"]
        or any(p["score"] is not None and p["score"] < 60 for p in phoneme_items)
    )

    error_type = safe_text(word_obj.get("error_type"))
    if error_type == "Omission":
        error_note = "This word was missing or not clearly spoken."
    elif error_type == "Mispronunciation":
        error_note = "This word was pronounced differently from the expected pronunciation."
    elif error_type in ["None", None] and any(p["score"] is not None and p["score"] < 60 for p in phoneme_items):
        low_phonemes = [p["phoneme"] for p in phoneme_items if p["score"] is not None and p["score"] < 60]
        error_note = f"Some sounds in this word need improvement, especially {format_phoneme_list(low_phonemes)}."
    elif error_type in ["None", None] and any(p["score"] is not None and p["score"] < 80 for p in phoneme_items):
        error_note = "This word is mostly understandable, but some sounds should be practiced more."
    else:
        error_note = "No major pronunciation error detected."

    return {
        "index": index,
        "word": safe_text(word_obj.get("word")),
        "raw_azure_score": raw_azure_score,
        "effective_score": effective_score,
        "color": word_obj.get("color") or ("red" if effective_score is not None and effective_score < 60 else "yellow" if effective_score is not None and effective_score < 80 else "green" if effective_score is not None else "gray"),
        "level": level_from_score(effective_score),
        "error_type": error_type,
        "error_note": error_note,
        "has_critical_issue": has_critical_issue,
        "phonemes": phoneme_items,
        "weak_phonemes": weak_phonemes,
    }


# ---------------------------------------------------------------------------
# Priority words & pronunciation summary
# ---------------------------------------------------------------------------

def make_priority_words(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    priority_candidates = [
        word for word in words
        if (word.get("effective_score") is not None and word.get("effective_score") < 80)
        or word.get("error_type") in ["Omission", "Mispronunciation"]
    ]

    def sort_key(word: Dict[str, Any]) -> Any:
        return (
            0 if word.get("error_type") == "Omission" else 1,
            0 if word.get("error_type") == "Mispronunciation" else 1,
            0 if word.get("color") == "red" else 1,
            word.get("effective_score") if word.get("effective_score") is not None else 100,
        )

    sorted_words = sorted(priority_candidates, key=sort_key)
    priority_list: List[Dict[str, Any]] = []
    for idx, word in enumerate(sorted_words, start=1):
        # Build detailed reason
        weak_phoneme_symbols = [
            p["phoneme"] for p in word.get("weak_phonemes", [])
            if p.get("phoneme")
        ]

        if word.get("error_type") == "Omission":
            reason = "This word was missing or not clearly spoken."
        elif word.get("error_type") == "Mispronunciation":
            reason = "This word was pronounced differently from the expected pronunciation."
        elif weak_phoneme_symbols:
            reason = f"Weak sounds: {format_phoneme_list(weak_phoneme_symbols)}."
        elif word.get("color") == "red":
            reason = "Low pronunciation accuracy."
        else:
            reason = "Needs further practice."

        practice_type = "phoneme_drill" if any(p["score"] is not None and p["score"] < 80 for p in word.get("phonemes", [])) else "word_drill"
        target_phonemes = [p["phoneme"] for p in word.get("phonemes", []) if p["score"] is not None and p["score"] < 80]

        priority_list.append({
            "priority": idx,
            "index": word.get("index"),
            "word": word.get("word"),
            "effective_score": word.get("effective_score"),
            "color": word.get("color"),
            "level": word.get("level"),
            "reason": reason,
            "target_phonemes": target_phonemes,
            "practice_type": practice_type,
        })
    return priority_list


def build_pronunciation_summary(words: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_word_count = len(words)
    weak_word_count = sum(1 for word in words if word.get("effective_score") is not None and word.get("effective_score") < 80)
    critical_word_count = sum(1 for word in words if word.get("has_critical_issue"))
    omitted_word_count = sum(1 for word in words if word.get("error_type") == "Omission")
    mispronounced_word_count = sum(1 for word in words if word.get("error_type") == "Mispronunciation")
    weak_words = [
        {
            "index": word.get("index"),
            "word": word.get("word"),
            "effective_score": word.get("effective_score"),
            "color": word.get("color"),
            "error_type": word.get("error_type"),
            "weak_phonemes": [p for p in word.get("weak_phonemes", [])],
        }
        for word in words
        if word.get("effective_score") is not None and word.get("effective_score") < 80
    ]
    return {
        "total_word_count": total_word_count,
        "weak_word_count": weak_word_count,
        "critical_word_count": critical_word_count,
        "omitted_word_count": omitted_word_count,
        "mispronounced_word_count": mispronounced_word_count,
        "priority_words": make_priority_words(words),
        "weak_words": weak_words,
    }


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------

def _build_too_short_rule_result(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Build a typed rule_result for answer_length.too_short."""
    word_count = metrics.get("word_count")
    expected_min = metrics.get("expected_min_words")
    length_ratio = None
    if word_count is not None and expected_min and expected_min > 0:
        length_ratio = round(word_count / expected_min, 2)

    score_caps: Dict[str, Any] = {}
    cap_keys = [
        ("coherence_max", "coherence_cap"),
        ("vocabulary_max", "lexical_range_cap"),
        ("grammar_max", "grammar_range_cap"),
    ]
    for out_key, in_key in cap_keys:
        val = safe_score(metrics.get(in_key))
        if val is not None:
            score_caps[out_key] = val

    return normalize_rule_result({
        "rule_id": "answer_length.too_short",
        "category": "length",
        "status": "triggered",
        "severity": "medium",
        "action": "score_with_penalty",
        "blocking": False,
        "target_criteria": ["coherence", "vocabulary", "grammar"],
        "message": safe_text(metrics.get("note")) or safe_text(metrics.get("length_note")) or "Answer is too short for the task.",
        "evidence": {
            "word_count": word_count,
            "expected_min_words": expected_min,
            "length_ratio": length_ratio,
            "question_type": metrics.get("question_type"),
            "duration_seconds": metrics.get("duration_seconds"),
        },
        "score_caps": score_caps,
        "penalties": [],
        "suggestion": "Expand your answer to include more details so coherence can be scored more accurately.",
    })


def build_validity(validity_payload: Any, metrics: Optional[Dict[str, Any]]) -> Any:
    """Return ValidityResult model object. Accepts model or dict."""
    if isinstance(validity_payload, ValidityResult):
        return validity_payload
    if isinstance(validity_payload, dict):
        return build_validity_result_from_payload(validity_payload)

    return build_validity_result_from_metrics(metrics)


# ---------------------------------------------------------------------------
# Topic relevance & language details
# ---------------------------------------------------------------------------

def build_topic_relevance(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(metadata, dict):
        return None

    topic = metadata.get("topic_relevance")
    if not isinstance(topic, dict):
        topic = {
            "score": metadata.get("topic_score"),
            "status": metadata.get("topic_status"),
            "question_text": metadata.get("question_text"),
            "topic_name": metadata.get("topic_name"),
            "matched_keywords": metadata.get("matched_keywords"),
            "missing_optional_keywords": metadata.get("missing_optional_keywords"),
            "note": metadata.get("topic_note"),
            "suggestion": metadata.get("topic_suggestion"),
        }

    if not any(topic.values()):
        return None

    return {
        "score": safe_score(topic.get("score")),
        "status": topic.get("status") or "not_evaluated",
        "question_text": topic.get("question_text") or "",
        "topic_name": topic.get("topic_name") or "",
        "matched_keywords": topic.get("matched_keywords") or [],
        "missing_optional_keywords": topic.get("missing_optional_keywords") or [],
        "note": topic.get("note") or "",
        "suggestion": topic.get("suggestion") or "",
    }


def build_language_details(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(metadata, dict):
        return []

    entries: List[Dict[str, Any]] = []
    if metadata.get("grammar_errors") is not None:
        entries.append({"type": "grammar", "detail": metadata.get("grammar_errors")})
    if metadata.get("vocabulary_notes") is not None:
        entries.append({"type": "vocabulary", "detail": metadata.get("vocabulary_notes")})
    if metadata.get("coherence_notes") is not None:
        entries.append({"type": "coherence", "detail": metadata.get("coherence_notes")})
    return entries


# ---------------------------------------------------------------------------
# Feedback (suggestions as object array)
# ---------------------------------------------------------------------------

def build_feedback(
    criteria: Dict[str, Any],
    length_metrics: Optional[Dict[str, Any]],
    pronunciation_summary: Dict[str, Any],
    priority_words: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    strengths: List[str] = []
    areas_to_improve: List[str] = []
    suggestions: List[Dict[str, Any]] = []

    pronunciation_score = criteria.get("pronunciation", {}).get("score")
    if pronunciation_score is not None and pronunciation_score >= 80:
        strengths.append("Pronunciation is generally clear.")
    elif pronunciation_score is not None:
        areas_to_improve.append("Pronunciation could be clearer in some words.")

    if length_metrics and safe_text(length_metrics.get("length_category")) == "too_short":
        areas_to_improve.append("The answer is too short for the task.")
        suggestions.append({
            "target": "coherence",
            "priority": 1,
            "message": "Expand your answer to include more details.",
        })

    if priority_words:
        suggestions.append({
            "target": "pronunciation",
            "priority": 2,
            "message": "Practice the priority words listed below.",
        })

    if metadata.get("llm_length_judgment"):
        llm_suggestion = metadata.get("llm_length_judgment", {}).get("suggestion")
        if llm_suggestion:
            suggestions.append({
                "target": "coherence",
                "priority": 3,
                "message": safe_text(llm_suggestion),
            })

    if criteria.get("grammar", {}).get("score") is not None and criteria["grammar"]["score"] >= 75:
        strengths.append("Grammar appears to be well controlled.")
    if criteria.get("vocabulary", {}).get("score") is not None and criteria["vocabulary"]["score"] >= 75:
        strengths.append("Vocabulary usage is appropriate for the task.")

    # Sort suggestions by priority
    suggestions.sort(key=lambda s: s.get("priority", 99))

    summary_fragments: List[str] = []
    if strengths:
        summary_fragments.append(" ".join(strengths))
    if areas_to_improve:
        summary_fragments.append(" ".join(areas_to_improve))
    if not summary_fragments:
        summary_fragments.append("The response is ready for review with detailed scoring available.")

    return {
        "summary": " ".join(summary_fragments),
        "strengths": strengths,
        "areas_to_improve": areas_to_improve,
        "suggestions": suggestions,
    }


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

def adapt_current_response_to_ui_response(current: Dict[str, Any]) -> Dict[str, Any]:
    metadata = current.get("metadata") or {}
    result = current.get("result") or {}
    if hasattr(result, "model_dump"):
        result = result.model_dump()
    elif hasattr(result, "dict"):
        result = result.dict()

    word_feedback_raw: List[Dict[str, Any]] = []
    if isinstance(result.get("word_feedback"), list):
        word_feedback_raw = result.get("word_feedback")
    elif isinstance(result.get("word_feedback"), dict):
        word_feedback_raw = [result.get("word_feedback")]

    word_feedback = [format_word_feedback(word, idx) for idx, word in enumerate(word_feedback_raw, start=1)]
    pronunciation_summary = build_pronunciation_summary(word_feedback)
    details_length = metadata.get("answer_length_metrics") if isinstance(metadata.get("answer_length_metrics"), dict) else {}

    validity_raw = current.get("validity")
    validity = build_validity(validity_raw, details_length)
    topic_relevance = build_topic_relevance(metadata)
    language_details = build_language_details(metadata)
    criteria = build_default_criteria(result)

    # Question context: current.question → metadata fallback
    q = current.get("question") if isinstance(current.get("question"), dict) else {}
    t = current.get("topic") if isinstance(current.get("topic"), dict) else {}
    q_id = q.get("question_id") or metadata.get("question_id")
    q_text = q.get("question_text") or metadata.get("question_text")
    q_type = q.get("question_type") or metadata.get("question_type")
    q_diff = q.get("difficulty_level") or metadata.get("difficulty_level")
    q_dur = q.get("duration_seconds") or metadata.get("duration_seconds")
    t_id = t.get("topic_id") or metadata.get("topic_id")
    t_name = t.get("topic_name") or metadata.get("topic_name")
    t_desc = t.get("topic_description") or metadata.get("topic_description")

    # Expected word counts from length metrics if available
    expected_min = details_length.get("expected_min_words") if isinstance(details_length, dict) else None
    expected_max = details_length.get("expected_max_words") if isinstance(details_length, dict) else None

    # Topic expected keywords from topic_relevance
    topic_matched = topic_relevance.get("matched_keywords") if topic_relevance else []
    topic_missing = topic_relevance.get("missing_optional_keywords") if topic_relevance else []

    # When validity has flags, override status to "error"
    validity_has_flags = bool(validity.flags) if hasattr(validity, "flags") else False
    response_status = "error" if validity_has_flags else current.get("status")
    response_error = None
    if validity_has_flags:
        first_flag = validity.flags[0] if validity.flags else None
        response_error = getattr(first_flag, "message", None) or "Response did not pass validity checks."

    ui_payload = {
        "status": response_status,
        "error": response_error or current.get("error"),
        "meta": {
            "audio_path": current.get("audio_path"),
            "mode": current.get("mode"),
        },
        "question": {
            "question_id": q_id,
            "question_text": q_text,
            "question_type": q_type,
            "difficulty_level": q_diff,
            "duration_seconds": q_dur,
            "expected_min_words": expected_min,
            "expected_max_words": expected_max,
            "topic": {
                "topic_id": t_id,
                "topic_name": t_name,
                "description": t_desc,
                "expected_keywords": topic_missing or [],
            },
        },
        "transcript": {
            "reference_text": current.get("reference_text") or (result.get("reference_text") if isinstance(result, dict) else None),
            "recognized_text": result.get("recognized_text"),
            "transcribed_text": metadata.get("transcription_text"),
            "original_transcript": metadata.get("original_transcript"),
            "corrected_transcript": metadata.get("corrected_transcript"),
        },
        "validity": validity,
        "scores": {
            "criteria": criteria,
        },
        "feedback": build_feedback(criteria, details_length, pronunciation_summary, pronunciation_summary.get("priority_words", []), metadata),
        "details": {
            "pronunciation": {
                "overall": result.get("overall") or {},
                "summary": pronunciation_summary,
                "word_feedback": word_feedback,
            },
            "length": {
                **(details_length or {}),
                "caps_applied": {
                    "coherence_max": safe_score(details_length.get("coherence_cap")),
                    "vocabulary_range_max": safe_score(details_length.get("lexical_range_cap")),
                    "grammar_range_max": safe_score(details_length.get("grammar_range_cap")),
                },
            },
            "topic_relevance": topic_relevance,
            "language": language_details,
        },
        "debug": {
            "raw_azure_result": result.get("raw_result"),
            "raw_llm_result": metadata.get("raw_llm_result"),
        },
    }
    return UIResponse.parse_obj(ui_payload).model_dump()
