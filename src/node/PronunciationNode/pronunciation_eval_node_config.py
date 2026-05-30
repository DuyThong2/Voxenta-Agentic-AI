"""
Pronunciation Evaluation Node using Azure Speech Pronunciation Assessment.

This node:
- Receives SpeakingInput from GraphState
- Calls Azure Speech Pronunciation Assessment
    - Returns formatted pronunciation result into GraphState
- unscripted: reference_text is None or empty
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

import azure.cognitiveservices.speech as speechsdk

from utils import load_root_dotenv
from node.state_models.pronunciation import (
    FormattedPronunciationResult,
    PhonemeFeedback,
    PronunciationAssessmentResult,
    WordFeedback,
)
from utils.pronunciation_formatter import format_pronunciation_api_response

load_root_dotenv()

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")


def score_to_color(score: Optional[float]) -> str:
    if score is None:
        return "gray"
    if score < 60:
        return "red"
    if score < 80:
        return "yellow"
    return "green"


def normalize_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None

    normalized = text.lower()
    normalized = re.sub(r"[^\w\s']", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or None


def build_speech_config() -> speechsdk.SpeechConfig:
    if not AZURE_SPEECH_KEY:
        raise RuntimeError("Missing AZURE_SPEECH_KEY in environment variables")

    if not AZURE_SPEECH_REGION:
        raise RuntimeError("Missing AZURE_SPEECH_REGION in environment variables")

    speech_config = speechsdk.SpeechConfig(
        subscription=AZURE_SPEECH_KEY,
        region=AZURE_SPEECH_REGION,
    )

    speech_config.output_format = speechsdk.OutputFormat.Detailed
    return speech_config


def extract_pronunciation_summary(data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    nbest = data.get("NBest", [])
    if not nbest:
        return {
            "accuracy_score": None,
            "fluency_score": None,
            "prosody_score": None,
            "pron_score": None,
            "completeness_score": None,
        }

    assessment = nbest[0].get("PronunciationAssessment", {})

    return {
        "accuracy_score": assessment.get("AccuracyScore"),
        "fluency_score": assessment.get("FluencyScore"),
        "prosody_score": assessment.get("ProsodyScore"),
        "pron_score": assessment.get("PronScore"),
        "completeness_score": assessment.get("CompletenessScore"),
    }


def extract_phoneme_feedback(phonemes: List[Dict[str, Any]]) -> List[PhonemeFeedback]:
    feedback: List[PhonemeFeedback] = []

    for phoneme in phonemes:
        assessment = phoneme.get("PronunciationAssessment", {})
        score = assessment.get("AccuracyScore")

        feedback.append(
            PhonemeFeedback(
                phoneme=phoneme.get("Phoneme", ""),
                accuracy_score=score,
                color=score_to_color(score),
            )
        )

    return feedback


def extract_word_feedback(data: Dict[str, Any]) -> List[WordFeedback]:
    nbest = data.get("NBest", [])
    if not nbest:
        return []

    words = nbest[0].get("Words", [])
    feedback: List[WordFeedback] = []

    for word in words:
        assessment = word.get("PronunciationAssessment", {})
        word_score = assessment.get("AccuracyScore")

        phoneme_items: List[Dict[str, Any]] = word.get("Phonemes", [])

        if not phoneme_items:
            for syllable in word.get("Syllables", []):
                phoneme_items.extend(syllable.get("Phonemes", []))

        feedback.append(
            WordFeedback(
                word=word.get("Word", ""),
                accuracy_score=word_score,
                error_type=assessment.get("ErrorType"),
                color=score_to_color(word_score),
                phonemes=extract_phoneme_feedback(phoneme_items),
            )
        )

    return feedback


def pronunciation_eval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function.

    Expected state:
    {
        "speaking_input": SpeakingInput(...)
    }

    Returns state with:
    {
        "pronunciation_result": PronunciationAssessmentResult(...),
        "status": "completed"
    }
    """

    speaking_input = state.get("speaking_input")

    if speaking_input is None:
        return {
            **state,
            "status": "error",
            "error": "speaking_input is required for pronunciation_eval_node",
        }

    audio_path = speaking_input.audio_path
    language = speaking_input.language or "en-US"
    
    # If reference_text is available, compare audio directly against it.
    # Otherwise, for unscripted mode, use corrected_transcript when available,
    # or fallback to raw transcribed_text.
    corrected_transcript = speaking_input.corrected_transcript

    if speaking_input.mode == "scripted":
        reference_text = normalize_text(speaking_input.reference_text or "") or ""
        if not reference_text:
            return {
                **state,
                "status": "error",
                "error": "reference_text is required for scripted pronunciation evaluation",
            }
    else:
        reference_text = corrected_transcript or speaking_input.transcribed_text or ""

    if not audio_path:
        return {
            **state,
            "status": "error",
            "error": "speaking_input.audio_path is required",
        }

    if not os.path.exists(audio_path):
        return {
            **state,
            "status": "error",
            "error": f"Audio file not found: {audio_path}",
        }

    try:
        speech_config = build_speech_config()

        audio_config = speechsdk.audio.AudioConfig(
            filename=audio_path
        )

        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            language=language,
            audio_config=audio_config,
        )

        pronunciation_config = speechsdk.PronunciationAssessmentConfig(
            reference_text=reference_text,
            grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
            granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
            enable_miscue=True if reference_text else False,
        )

        # Prosody gives intonation/stress/rhythm related feedback.
        # You can comment this out if you only want basic pronunciation score.
        pronunciation_config.enable_prosody_assessment()

        pronunciation_config.apply_to(recognizer)

        result = recognizer.recognize_once()

        raw_json = result.properties.get(
            speechsdk.PropertyId.SpeechServiceResponse_JsonResult
        )

        if result.reason != speechsdk.ResultReason.RecognizedSpeech:
            cancellation_details = None

            if result.reason == speechsdk.ResultReason.Canceled:
                cancellation = speechsdk.CancellationDetails.from_result(result)
                cancellation_details = {
                    "reason": str(cancellation.reason),
                    "error_details": cancellation.error_details,
                }

            return {
                **state,
                "status": "error",
                "error": f"Azure speech recognition failed: {result.reason}",
                "metadata": {
                    **state.get("metadata", {}),
                    "raw_azure_response": raw_json,
                    "cancellation_details": cancellation_details,
                },
            }

        data = json.loads(raw_json) if raw_json else {}
        summary = extract_pronunciation_summary(data)

        pronunciation_result = PronunciationAssessmentResult(
            recognized_text=result.text,
            accuracy_score=summary.get("accuracy_score"),
            fluency_score=summary.get("fluency_score"),
            prosody_score=summary.get("prosody_score"),
            pron_score=summary.get("pron_score"),
            completeness_score=summary.get("completeness_score"),
            word_feedback=extract_word_feedback(data),
            raw_result=data,
        )

        formatted_result = format_pronunciation_api_response(
            pronunciation_result,
            mode=speaking_input.mode,
            reference_text=reference_text if reference_text else None,
            include_raw=False,
        )

        return {
            **state,
            "pronunciation_result": formatted_result,
            "status": "completed",
            "error": None,
        }

    except Exception as exc:
        return {
            **state,
            "status": "error",
            "error": str(exc),
        }