"""Start node: transcribe audio and normalize text before pronunciation evaluation."""

import os
import re
from typing import Optional

import azure.cognitiveservices.speech as speechsdk

from node.GraphState import GraphState
from node.state_models import SpeakingInput
from utils import load_root_dotenv


def build_speech_config(language: str) -> speechsdk.SpeechConfig:
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION")

    if not speech_key:
        raise RuntimeError("Missing AZURE_SPEECH_KEY in environment variables")

    if not speech_region:
        raise RuntimeError("Missing AZURE_SPEECH_REGION in environment variables")

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.output_format = speechsdk.OutputFormat.Simple
    speech_config.speech_recognition_language = language
    return speech_config


def normalize_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None

    normalized = text.lower()
    normalized = re.sub(r"[^\w\s']", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or None


def start_node(state: GraphState) -> dict:
    speaking_input = state.get("speaking_input")

    if speaking_input is None:
        return {
            **state,
            "status": "error",
            "error": "speaking_input is required for start_node",
        }

    audio_path = speaking_input.audio_path
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

    load_root_dotenv()

    try:
        speech_config = build_speech_config(speaking_input.language)
        audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )

        result = recognizer.recognize_once()
        transcript = result.text if result.reason == speechsdk.ResultReason.RecognizedSpeech else None

        if speaking_input.mode == "unscripted" and not transcript:
            cancellation = None
            if result.reason == speechsdk.ResultReason.Canceled:
                cancellation = speechsdk.CancellationDetails.from_result(result)
            return {
                **state,
                "status": "error",
                "error": "Audio transcription failed for unscripted mode",
                "metadata": {
                    **state.get("metadata", {}),
                    "recognition_reason": str(result.reason),
                    "cancellation_details": {
                        "reason": str(cancellation.reason) if cancellation else None,
                        "error_details": cancellation.error_details if cancellation else None,
                    },
                },
            }

        normalized_transcript = normalize_text(transcript)

        if speaking_input.mode == "scripted":
            # For scripted mode, reference_text is authoritative and should be normalized.
            speaking_input.reference_text = normalize_text(speaking_input.reference_text)

        speaking_input.transcribed_text = transcript

        return {
            **state,
            "speaking_input": speaking_input,
            "status": "processing",
            "metadata": {
                **state.get("metadata", {}),
                "transcription_text": transcript,
            },
        }

    except Exception as exc:
        return {
            **state,
            "status": "error",
            "error": str(exc),
        }
