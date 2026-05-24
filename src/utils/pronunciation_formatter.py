"""
Utilities for formatting Azure Pronunciation Assessment results
into frontend-friendly API responses.
"""

from typing import Any, Dict, List, Optional

from node.state_models import PronunciationAssessmentResult, WordFeedback, PhonemeFeedback


RED_THRESHOLD = 60
YELLOW_THRESHOLD = 80


def score_color(score: Optional[float]) -> str:
    if score is None:
        return "gray"
    if score < RED_THRESHOLD:
        return "red"
    if score < YELLOW_THRESHOLD:
        return "yellow"
    return "green"


def score_level(score: Optional[float]) -> str:
    if score is None:
        return "unknown"
    if score < RED_THRESHOLD:
        return "needs_improvement"
    if score < YELLOW_THRESHOLD:
        return "fair"
    return "good"


def round_score(score: Optional[float]) -> Optional[float]:
    if score is None:
        return None
    return round(float(score), 1)


def get_lowest_phoneme_score(phonemes: List[PhonemeFeedback]) -> Optional[float]:
    scores = [
        phoneme.accuracy_score
        for phoneme in phonemes
        if phoneme.accuracy_score is not None
    ]

    if not scores:
        return None

    return min(scores)


def get_effective_word_score(word: WordFeedback) -> Optional[float]:
    """
    Azure word score can sometimes look high even when one phoneme is very weak.
    For frontend warning, use the lowest score between word score and phoneme score.

    Examples:
    - word bus = 94, phoneme /s/ = 0 -> effective score = 0
    - word school = Omission -> effective score = 0
    """

    if word.error_type in ["Omission", "Insertion"]:
        return 0

    word_score = word.accuracy_score
    lowest_phoneme = get_lowest_phoneme_score(word.phonemes)

    if word_score is None:
        return lowest_phoneme

    if lowest_phoneme is None:
        return word_score

    return min(word_score, lowest_phoneme)


def explain_error_type(error_type: Optional[str]) -> str:
    mapping = {
        "None": "No major pronunciation error detected.",
        "Mispronunciation": "This word was pronounced differently from the expected pronunciation.",
        "Omission": "This word was missing or not clearly spoken.",
        "Insertion": "An extra word or sound may have been added.",
        "UnexpectedBreak": "There was an unnatural pause before or inside this word.",
        "MissingBreak": "A natural pause may be missing.",
        "Monotone": "The intonation sounds flat or lacks variation.",
    }

    if not error_type:
        return "No error type provided."

    return mapping.get(error_type, f"Detected issue: {error_type}")


def explain_phoneme_issue(phoneme: str, score: Optional[float]) -> str:
    if score is None:
        return f"The sound /{phoneme}/ could not be scored."

    if score < RED_THRESHOLD:
        return f"The sound /{phoneme}/ needs clear improvement."
    if score < YELLOW_THRESHOLD:
        return f"The sound /{phoneme}/ is understandable but should be practiced more."

    return f"The sound /{phoneme}/ is good."


def format_phoneme_feedback(phoneme: PhonemeFeedback) -> Dict[str, Any]:
    score = round_score(phoneme.accuracy_score)

    return {
        "phoneme": phoneme.phoneme,
        "score": score,
        "color": score_color(score),
        "level": score_level(score),
        "note": explain_phoneme_issue(phoneme.phoneme, score),
    }


def format_word_feedback(word: WordFeedback) -> Dict[str, Any]:
    azure_word_score = round_score(word.accuracy_score)
    effective_score = round_score(get_effective_word_score(word))

    formatted_phonemes = [
        format_phoneme_feedback(phoneme)
        for phoneme in word.phonemes
    ]

    weak_phonemes = [
        phoneme
        for phoneme in formatted_phonemes
        if phoneme["score"] is not None and phoneme["score"] < YELLOW_THRESHOLD
    ]

    has_critical_issue = (
        word.error_type in ["Omission", "Insertion", "Mispronunciation"]
        or any(
            phoneme["score"] is not None and phoneme["score"] < RED_THRESHOLD
            for phoneme in formatted_phonemes
        )
    )

    return {
        "word": word.word,
        "azure_score": azure_word_score,
        "effective_score": effective_score,
        "color": score_color(effective_score),
        "level": score_level(effective_score),
        "error_type": word.error_type,
        "error_note": explain_error_type(word.error_type),
        "has_critical_issue": has_critical_issue,
        "phonemes": formatted_phonemes,
        "weak_phonemes": weak_phonemes,
    }


def build_correction_summary(words: List[Dict[str, Any]]) -> Dict[str, Any]:
    weak_words = [
        word
        for word in words
        if word["effective_score"] is not None and word["effective_score"] < YELLOW_THRESHOLD
    ]

    critical_words = [
        word
        for word in words
        if word["has_critical_issue"]
    ]

    omitted_words = [
        word
        for word in words
        if word["error_type"] == "Omission"
    ]

    mispronounced_words = [
        word
        for word in words
        if word["error_type"] == "Mispronunciation"
    ]

    return {
        "weak_word_count": len(weak_words),
        "critical_word_count": len(critical_words),
        "omitted_word_count": len(omitted_words),
        "mispronounced_word_count": len(mispronounced_words),
        "weak_words": [
            {
                "word": word["word"],
                "effective_score": word["effective_score"],
                "color": word["color"],
                "error_type": word["error_type"],
                "weak_phonemes": word["weak_phonemes"],
            }
            for word in weak_words
        ],
    }


def build_ielts_like_scores(result: PronunciationAssessmentResult) -> Dict[str, Any]:
    """
    This is not official IELTS scoring.
    It only formats the system's internal criteria in an IELTS-like structure.

    Future expansion:
    - pronunciation: Azure
    - fluency_and_coherence: Azure fluency + LLM coherence
    - lexical_resource: LLM vocabulary
    - grammatical_range_and_accuracy: LLM grammar
    """

    return {
        "pronunciation": {
            "score": round_score(result.pron_score or result.accuracy_score),
            "subscores": {
                "accuracy": round_score(result.accuracy_score),
                "prosody": round_score(result.prosody_score),
            },
            "note": "Based on pronunciation accuracy and prosody from speech assessment.",
        },
        "fluency_and_coherence": {
            "score": round_score(result.fluency_score),
            "subscores": {
                "fluency": round_score(result.fluency_score),
                "coherence": None,
            },
            "note": "Current version only includes fluency from audio. Coherence will be added from transcript analysis.",
        },
        "lexical_resource": {
            "score": None,
            "subscores": {},
            "note": "Not evaluated yet. This will be evaluated from transcript by the LLM module.",
        },
        "grammatical_range_and_accuracy": {
            "score": None,
            "subscores": {},
            "note": "Not evaluated yet. This will be evaluated from transcript by the LLM module.",
        },
    }


def format_pronunciation_api_response(
    result: PronunciationAssessmentResult,
    *,
    mode: str,
    reference_text: Optional[str] = None,
    include_raw: bool = False,
) -> Dict[str, Any]:
    """
    Format pronunciation result for frontend.

    Frontend should use:
    - criteria.pronunciation
    - criteria.fluency_and_coherence
    - correction_summary.weak_words
    - word_feedback
    """

    words = [
        format_word_feedback(word)
        for word in result.word_feedback
    ]

    response = {
        "recognized_text": result.recognized_text,
        "reference_text": reference_text,
        "mode": mode,
        "overall": {
            "pronunciation_score": round_score(result.pron_score),
            "accuracy_score": round_score(result.accuracy_score),
            "fluency_score": round_score(result.fluency_score),
            "prosody_score": round_score(result.prosody_score),
            "completeness_score": (
                round_score(result.completeness_score)
                if mode == "scripted"
                else None
            ),
        },
        "criteria": build_ielts_like_scores(result),
        "correction_summary": build_correction_summary(words),
        "word_feedback": words,
        "notes": [
            "Scores are generated by speech assessment and should be interpreted as learning feedback, not final exam grading.",
            "Word color uses an effective score that considers both word-level and phoneme-level errors.",
            "If a word has a very weak phoneme, the word may be highlighted even when Azure word score is high.",
        ],
    }

    if include_raw:
        response["raw_result"] = result.raw_result

    return response