"""Realtime pronunciation scoring -- adapted from (not a copy of)
node/evalGraph/PronunciationNode/pronunciation_eval_node_config.py: reuses the same
speechsdk.PronunciationAssessmentConfig call pattern, but skips the SCRIPTED/UNSCRIPTED
reference-text-consensus branch (no LLM round-trip here) and uses the turn's own transcript
directly as reference_text -- good enough for a fast mid-session pronunciation snapshot,
not a substitute for the full end-of-session eval.

audio_path comes from PracticeAttemptConnection._write_turn_wav -- it buffers the same raw
PCM16 bytes already being forwarded to Voice Live (no second upload, no /turns/archive
-equivalent needed, see task/implement/11-toi-uu-dung-de.md "Turn-recording trong phiên
realtime") and writes them to a temp WAV per turn, deleted right after this graph runs. Still
returns None gracefully (see below) for the rare case the buffer was empty (e.g. a turn with
literally no captured audio).
"""

import json
import logging
import os
import threading
from typing import Any, Dict, List

import azure.cognitiveservices.speech as speechsdk

from utils.speech_client import build_speech_config, describe_no_match

logger = logging.getLogger(__name__)


def pronunciation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    audio_path = state.get("audio_path")
    if not audio_path or not os.path.exists(audio_path):
        logger.info("[realtime_correction:pronunciation] no audio_path, skipping")
        return {"pronunciation_result": None}

    transcript = (state.get("transcript") or "").strip()
    language = state.get("language") or "en-US"
    if not transcript:
        return {"pronunciation_result": None}

    try:
        speech_config = build_speech_config(language, output_format=speechsdk.OutputFormat.Detailed)
        audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, language=language, audio_config=audio_config,
        )
        pronunciation_config = speechsdk.PronunciationAssessmentConfig(
            reference_text=transcript,
            grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
            granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
            enable_miscue=True,
        )
        pronunciation_config.apply_to(recognizer)

        segments_data: List[Dict[str, Any]] = []
        done = threading.Event()
        canceled_error = None

        def on_recognized(evt) -> None:
            seg_result = evt.result
            if seg_result.reason != speechsdk.ResultReason.RecognizedSpeech or not seg_result.text:
                if seg_result.reason == speechsdk.ResultReason.NoMatch:
                    logger.warning(
                        "[realtime_correction:pronunciation] NoMatch reason=%s",
                        describe_no_match(seg_result),
                    )
                return
            seg_raw_json = seg_result.properties.get(
                speechsdk.PropertyId.SpeechServiceResponse_JsonResult
            )
            if seg_raw_json:
                segments_data.append(json.loads(seg_raw_json))

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

        recognizer.start_continuous_recognition()
        try:
            done.wait(timeout=15.0)
        finally:
            recognizer.stop_continuous_recognition()

        if canceled_error or not segments_data:
            if canceled_error:
                logger.warning("[realtime_correction:pronunciation] canceled: %s", canceled_error)
            return {"pronunciation_result": None}

        nbest = segments_data[0].get("NBest", [{}])[0]
        assessment = nbest.get("PronunciationAssessment", {})
        return {
            "pronunciation_result": {
                "accuracy_score": assessment.get("AccuracyScore"),
                "fluency_score": assessment.get("FluencyScore"),
                "completeness_score": assessment.get("CompletenessScore"),
                "pron_score": assessment.get("PronScore"),
            }
        }
    except Exception:
        logger.exception("[realtime_correction:pronunciation] failed")
        return {"pronunciation_result": None}
