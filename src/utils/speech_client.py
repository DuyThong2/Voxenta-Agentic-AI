import os
import re
import threading
from typing import List, NamedTuple, Optional

import azure.cognitiveservices.speech as speechsdk

from utils import load_root_dotenv

# Candidate languages for continuous Language Identification during archive
# re-transcription: the exam's target language (English) plus the student
# population's native language (Vietnamese), so a student code-switching
# mid-answer is recognized correctly instead of mangled as garbled English.
# Azure caps continuous LID at 4 candidates; only 2 are used, leaving
# headroom if a third language is ever needed.
CODE_SWITCH_CANDIDATE_LANGUAGES = ["en-US", "vi-VN"]

# Matches the "[XX: text]" wrapper transcribe() adds around a non-target-
# language segment (see _wrap_non_target_segment below). Used by callers
# that need to exclude code-switched text from word/sentence counts.
NON_TARGET_SEGMENT_PATTERN = re.compile(r"\[[A-Z]{2}: [^\]]*\]")
NON_TARGET_SEGMENT_CAPTURE_PATTERN = re.compile(r"\[[A-Z]{2}: ([^\]]*)\]")


def strip_non_target_segments(text: Optional[str]) -> str:
    """Remove "[XX: ...]" code-switch wrapper segments, leaving only the
    target-language text. Used wherever word/sentence counts must reflect
    only the language actually being graded (see answer_length_analysis_node).
    """
    if not text:
        return ""
    return re.sub(NON_TARGET_SEGMENT_PATTERN, " ", text)


def extract_non_target_segments(text: Optional[str]) -> List[str]:
    """Return just the inner text of each "[XX: ...]" code-switch wrapper
    segment, for computing codeSwitchingRatio (word count of non-target
    segments / total word count)."""
    if not text:
        return []
    return [match.group(1) for match in NON_TARGET_SEGMENT_CAPTURE_PATTERN.finditer(text)]


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


def _build_auto_detect_config(target_language: str) -> speechsdk.languageconfig.AutoDetectSourceLanguageConfig:
    candidates = [target_language] + [
        lang for lang in CODE_SWITCH_CANDIDATE_LANGUAGES if lang.lower() != target_language.lower()
    ]
    # Azure continuous Language Identification caps out at 4 candidates.
    return speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=candidates[:4])


def build_recognizer(audio_path: str, language: str) -> speechsdk.SpeechRecognizer:
    speech_config = build_speech_config(language)
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
    return speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
        auto_detect_source_language_config=_build_auto_detect_config(language),
    )


class _TranscribedSegment(NamedTuple):
    text: str
    language: str


def _wrap_non_target_segment(text: str, detected_language: str) -> str:
    lang_code = (detected_language or "").split("-")[0].upper() or "XX"
    return f"[{lang_code}: {text}]"


def _probe_audio_duration_seconds(audio_path: str) -> Optional[float]:
    """Real duration of the downloaded WAV, used to scale how long we wait
    for continuous recognition to finish (near-instant recognize_once() used
    to make this moot; continuous recognition runs proportionally to real
    audio length)."""
    import wave

    try:
        with wave.open(audio_path, "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            if rate <= 0:
                return None
            return frames / float(rate)
    except (wave.Error, OSError, EOFError):
        return None


def transcribe(audio_path: str, language: str) -> Optional[str]:
    """Transcribe the full audio file.

    Uses continuous recognition (start_continuous_recognition), NOT
    recognize_once() -- recognize_once() stops after the first detected
    utterance/silence and returns only that segment, silently truncating any
    answer containing a natural pause between sentences (normal for anything
    beyond a one-breath answer). Segments continuous Language Identification
    detects as a language other than `language` are wrapped as "[XX: text]"
    (e.g. "[VI: ...]") rather than transcribed as if they were the target
    language -- callers that need student-only target-language text should
    strip these via strip_non_target_segments() before counting words.
    """
    recognizer = build_recognizer(audio_path, language)
    segments: List[_TranscribedSegment] = []
    done = threading.Event()
    canceled_error: Optional[str] = None
    target_prefix = language.split("-")[0].lower()

    def on_recognized(evt) -> None:
        result = evt.result
        if result.reason != speechsdk.ResultReason.RecognizedSpeech or not result.text:
            return
        detected = speechsdk.AutoDetectSourceLanguageResult(result).language or language
        segments.append(_TranscribedSegment(text=result.text, language=detected))

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
        raise RuntimeError(f"Azure continuous recognition canceled: {canceled_error}")

    if not segments:
        return None

    parts = [
        seg.text if seg.language.lower().startswith(target_prefix) else _wrap_non_target_segment(seg.text, seg.language)
        for seg in segments
    ]
    return " ".join(parts)
