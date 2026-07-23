"""
Pronunciation Evaluation Node using Azure Speech Pronunciation Assessment.

This node:
- Receives SpeakingInput from GraphState
- Calls Azure Speech Pronunciation Assessment
    - Returns formatted pronunciation result into GraphState
- unscripted: reference_text is None or empty
"""

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

import azure.cognitiveservices.speech as speechsdk

from schemas.enums import ScoreColor, SpeakingMode
from node.state_models.pronunciation import (
    FormattedPronunciationResult,
    PhonemeFeedback,
    PronunciationAssessmentResult,
    WordFeedback,
)
from node.evalGraph.PronunciationNode.pronunciation_node_helper import format_pronunciation_api_response
from node.evalGraph.PronunciationNode.pronunciation_reference_helper import (
    build_pronunciation_reference_consensus,
)
from utils import load_root_dotenv
from utils.confidence_utils import compute_alignment_confidence
from utils.speech_client import build_speech_config, normalize_text, _probe_audio_duration_seconds

load_root_dotenv()

logger = logging.getLogger(__name__)


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
                color=ScoreColor.from_score(score),
            )
        )

    return feedback


def _word_count(data: Dict[str, Any]) -> int:
    nbest = data.get("NBest", [])
    if not nbest:
        return 0
    return len(nbest[0].get("Words", []))


def merge_pronunciation_summaries(segments_data: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Combine one PronunciationAssessment summary per continuous-recognition
    segment into a single overall summary, weighted by each segment's word
    count (a short segment's score shouldn't count as much as a long one)."""
    keys = ["accuracy_score", "fluency_score", "prosody_score", "pron_score", "completeness_score"]
    merged: Dict[str, Optional[float]] = {key: None for key in keys}

    weighted_summaries = [
        (extract_pronunciation_summary(data), max(_word_count(data), 1))
        for data in segments_data
    ]

    for key in keys:
        pairs = [(summary.get(key), weight) for summary, weight in weighted_summaries if summary.get(key) is not None]
        if not pairs:
            continue
        total_weight = sum(weight for _, weight in pairs)
        merged[key] = sum(value * weight for value, weight in pairs) / total_weight

    return merged


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
                color=ScoreColor.from_score(word_score),
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
        return {"metadata": {"pronunciation_error": "speaking_input is required for pronunciation_eval_node"}}

    answer_id = getattr(speaking_input, "answer_id", None)
    turn_order = (state.get("metadata") or {}).get("turn_order")

    audio_path = speaking_input.audio_path
    language = speaking_input.language or "en-US"
    c_ref: Optional[float] = None
    
    # If reference_text is available, compare audio directly against it.
    # Otherwise, for unscripted mode, use Azure's own transcribed_text.
    if speaking_input.mode == SpeakingMode.SCRIPTED:
        reference_text = normalize_text(speaking_input.reference_text or "") or ""
        if not reference_text:
            return {"metadata": {"pronunciation_error": "reference_text is required for scripted pronunciation evaluation"}}
    else:
        raw_transcript = speaking_input.transcribed_text or ""
        if raw_transcript:
            try:
                reference_text, c_ref = build_pronunciation_reference_consensus(
                    raw_transcript,
                    speaking_input.question,
                )
                logger.info(
                    "[eval:pronunciation] built reference from raw transcript answer_id=%s turn=%s changed=%s c_ref=%s",
                    answer_id, turn_order, reference_text != raw_transcript, c_ref,
                )
            except Exception:
                # This reference text only feeds Azure's forced-alignment below -- validity,
                # grammar/lexical/coherence eval and UI display all keep reading the raw
                # transcribed_text untouched, so a failure here must not abort the turn.
                logger.exception(
                    "[eval:pronunciation] reference correction failed, falling back to raw transcript answer_id=%s turn=%s",
                    answer_id, turn_order,
                )
                reference_text = raw_transcript
        else:
            reference_text = ""

    if not audio_path:
        return {"metadata": {"pronunciation_error": "speaking_input.audio_path is required"}}

    if not os.path.exists(audio_path):
        return {"metadata": {"pronunciation_error": f"Audio file not found: {audio_path}"}}

    logger.info("[eval:pronunciation] calling Azure pronunciation assessment answer_id=%s turn=%s", answer_id, turn_order)

    try:
        speech_config = build_speech_config(
            language,
            output_format=speechsdk.OutputFormat.Detailed,
        )

        audio_config = speechsdk.audio.AudioConfig(
            filename=audio_path
        )

        # Fixed language=, NOT auto_detect_source_language_config: tried combining
        # PronunciationAssessmentConfig with auto-detect to stop code-switched Vietnamese
        # words (e.g. "banh mi") from being silently dropped from Words[] -- confirmed live
        # this broke detailed phoneme assessment entirely (word_feedback came back empty).
        # Detailed phoneme scoring for the English content is the core of this feature and
        # matters far more than a code-switched word appearing in the word-tag list, so this
        # stays a fixed single-language recognizer. The DISPLAY-only fix for code-switched
        # text is instead: show speaking_input.transcribed_text (Azure's own transcription,
        # already auto-detect + language-tagged, has the Vietnamese) as the turn's summary
        # line, not a transcript rebuilt from this recognizer's Words[] -- see
        # WordFeedbackText.tsx / exam_event_builder.display_transcript.
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

        # Continuous recognition, not recognize_once() -- recognize_once()
        # stops after the first detected utterance/silence and returns only
        # that segment, so any answer with a natural pause between sentences
        # has everything after the pause silently missing from this
        # assessment. Because reference_text (built from the turn's own
        # transcript) still contains those later words, Azure's miscue
        # alignment then flags them as a block of "Omission" (bright red)
        # even though the student really did say them -- this is exactly
        # what was showing up as red word spans on otherwise well-scored
        # answers. Continuous recognition + a reference_text is Microsoft's
        # own documented pattern for multi-sentence/paused audio (see
        # speech_client.transcribe(), which already uses this same
        # approach) -- each recognized segment's own PronunciationAssessment
        # JSON is accumulated here and merged below.
        segments_data: List[Dict[str, Any]] = []
        recognized_text_parts: List[str] = []
        done = threading.Event()
        canceled_error: Optional[str] = None

        def on_recognized(evt) -> None:
            seg_result = evt.result
            if seg_result.reason != speechsdk.ResultReason.RecognizedSpeech or not seg_result.text:
                return
            seg_raw_json = seg_result.properties.get(
                speechsdk.PropertyId.SpeechServiceResponse_JsonResult
            )
            if seg_raw_json:
                segments_data.append(json.loads(seg_raw_json))
            recognized_text_parts.append(seg_result.text)

        def on_canceled(evt) -> None:
            nonlocal canceled_error
            if evt.reason == speechsdk.CancellationReason.Error:
                canceled_error = evt.error_details
            done.set()

        def on_stopped(evt) -> None:
            done.set()

        recognizer.recognized.connect(on_recognized)
        recognizer.session_stopped.connect(on_stopped)
        recognizer.canceled.connect(on_canceled)

        audio_duration = _probe_audio_duration_seconds(audio_path)
        timeout_seconds = max(30.0, (audio_duration or 0.0) * 1.5)

        recognizer.start_continuous_recognition()
        try:
            done.wait(timeout=timeout_seconds)
        finally:
            recognizer.stop_continuous_recognition()

        if canceled_error:
            return {"metadata": {"pronunciation_error": f"Azure continuous recognition canceled: {canceled_error}"}}

        if not segments_data:
            return {"metadata": {"pronunciation_error": "Azure speech recognition returned no recognized segments"}}

        summary = merge_pronunciation_summaries(segments_data)
        merged_word_feedback: List[WordFeedback] = []
        for seg_data in segments_data:
            merged_word_feedback.extend(extract_word_feedback(seg_data))

        pronunciation_result = PronunciationAssessmentResult(
            recognized_text=" ".join(recognized_text_parts),
            accuracy_score=summary.get("accuracy_score"),
            fluency_score=summary.get("fluency_score"),
            prosody_score=summary.get("prosody_score"),
            pron_score=summary.get("pron_score"),
            completeness_score=summary.get("completeness_score"),
            word_feedback=merged_word_feedback,
            raw_result={"segments": segments_data},
        )

        formatted_result = format_pronunciation_api_response(
            pronunciation_result,
            mode=speaking_input.mode,
            reference_text=reference_text if reference_text else None,
            include_raw=False,
            criteria_frameworks=speaking_input.criteria_frameworks,
        )
        c_align = compute_alignment_confidence(segments_data, reference_text)

        logger.info(
            "[eval:pronunciation] done answer_id=%s turn=%s pron_score=%s",
            answer_id, turn_order, pronunciation_result.pron_score,
        )

        return {
            "pronunciation_result": formatted_result,
            "metadata": {
                "c_ref": c_ref,
                "c_align": c_align,
            },
        }

    except Exception as exc:
        logger.exception("[eval:pronunciation] failed answer_id=%s turn=%s", answer_id, turn_order)
        return {"metadata": {"pronunciation_error": str(exc)}}
