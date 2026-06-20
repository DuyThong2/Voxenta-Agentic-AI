import os
import re
from typing import Optional

import azure.cognitiveservices.speech as speechsdk

from utils import load_root_dotenv


def normalize_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None

    normalized = text.lower()
    normalized = re.sub(r"[^\w\s']", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or None


def build_speech_config(
    language: str,
    *,
    output_format: speechsdk.OutputFormat = speechsdk.OutputFormat.Simple,
) -> speechsdk.SpeechConfig:
    load_root_dotenv()

    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION")

    if not speech_key:
        raise RuntimeError("Missing AZURE_SPEECH_KEY in environment variables")

    if not speech_region:
        raise RuntimeError("Missing AZURE_SPEECH_REGION in environment variables")

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.output_format = output_format
    speech_config.speech_recognition_language = language
    return speech_config


def build_recognizer(audio_path: str, language: str) -> speechsdk.SpeechRecognizer:
    speech_config = build_speech_config(language)
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
    return speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
    )


def transcribe(audio_path: str, language: str) -> Optional[str]:
    recognizer = build_recognizer(audio_path, language)
    result = recognizer.recognize_once()

    if result.reason != speechsdk.ResultReason.RecognizedSpeech:
        return None

    return result.text
