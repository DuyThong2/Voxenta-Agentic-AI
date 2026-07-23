import json
import logging
import os
import re
import threading
import unicodedata
import wave
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

import azure.cognitiveservices.speech as speechsdk

from utils import load_root_dotenv

logger = logging.getLogger(__name__)

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


def unwrap_language_tags(text: Optional[str]) -> str:
    """Remove the "[XX: ...]" wrapper punctuation while keeping the text inside it in
    place, e.g. "I go to [VI: đại học] every day" -> "I go to đại học every day". Used
    for DISPLAY (what the student actually said, read naturally) -- unlike
    strip_non_target_segments (which deletes the wrapped text entirely, for word/sentence
    counts) or extract_non_target_segments (which isolates only the wrapped text, for
    codeSwitchingRatio), this keeps the full sentence intact and just drops the
    internal "[XX: ]" bookkeeping markup the reader has no reason to see."""
    if not text:
        return ""
    return re.sub(NON_TARGET_SEGMENT_CAPTURE_PATTERN, r"\1", text)


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
    confidence: Optional[float]


def _segment_confidence(result) -> Optional[float]:
    """Best-effort NBest[0].Confidence from Azure's detailed JSON result, the
    same per-segment confidence Azure itself uses for Pronunciation
    Assessment. Returns None (rather than raising) if the property is
    missing/malformed -- confidence is a diagnostic signal, not something
    that should ever fail a transcription."""
    try:
        raw_json = result.properties.get(speechsdk.PropertyId.SpeechServiceResponse_JsonResult)
        if not raw_json:
            return None
        nbest = json.loads(raw_json).get("NBest", [])
        if not nbest:
            return None
        confidence = nbest[0].get("Confidence")
        return float(confidence) if confidence is not None else None
    except (ValueError, TypeError, KeyError):
        return None


def _wrap_non_target_segment(text: str, detected_language: str) -> str:
    lang_code = (detected_language or "").split("-")[0].upper() or "XX"
    return f"[{lang_code}: {text}]"


def _probe_audio_duration_seconds(audio_path: str) -> Optional[float]:
    """Real duration of the downloaded WAV, used to scale how long we wait
    for continuous recognition to finish (near-instant recognize_once() used
    to make this moot; continuous recognition runs proportionally to real
    audio length)."""
    try:
        with wave.open(audio_path, "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            if rate <= 0:
                return None
            return frames / float(rate)
    except (wave.Error, OSError, EOFError):
        return None


def compute_silence_ratio(audio_path: str) -> Optional[float]:
    """Fraction of the recording that's near-silence, measured directly from
    the WAV waveform's short-time RMS energy -- independent of ASR. This is
    the one audio-quality signal that's always available regardless of
    transcription source: when the live Voice-Live transcript is used
    (see start_node_config.py), there's no per-segment ASR confidence at all,
    so exam_event_builder._compute_audio_quality would otherwise have nothing
    to fall back to and audioQuality would always come through as null/0."""
    try:
        with wave.open(audio_path, "rb") as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            raw = wav_file.readframes(wav_file.getnframes())
    except (wave.Error, OSError, EOFError):
        return None

    if not raw or sample_width != 2 or frame_rate <= 0:
        return None

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    window_size = max(1, int(frame_rate * 0.03))  # ~30ms windows
    n_windows = samples.size // window_size
    if n_windows == 0:
        return None

    windows = samples[: n_windows * window_size].reshape(n_windows, window_size)
    rms = np.sqrt(np.mean(windows ** 2, axis=1))

    # Relative threshold (vs. this clip's own loud passages) rather than a
    # fixed absolute level, since recording gain varies per device/mic.
    peak_rms = float(np.percentile(rms, 95))
    if peak_rms <= 0:
        return 1.0

    threshold = peak_rms * 0.05
    silence_windows = int(np.sum(rms < threshold))

    return silence_windows / n_windows


def compute_snr_db(audio_path: str) -> Optional[float]:
    """Ước lượng segmental SNR (dB) bằng cùng window/ngưỡng với silence ratio.

    Đây không phải WADA-SNR. Cách đo minh bạch này xem các window dưới ngưỡng
    RMS tương đối là nhiễu nền và các window còn lại là tín hiệu lời nói.
    """
    try:
        with wave.open(audio_path, "rb") as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            raw = wav_file.readframes(wav_file.getnframes())
    except (wave.Error, OSError, EOFError):
        return None

    if not raw or sample_width != 2 or frame_rate <= 0:
        return None

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    window_size = max(1, int(frame_rate * 0.03))
    n_windows = samples.size // window_size
    if n_windows == 0:
        return None

    windows = samples[: n_windows * window_size].reshape(n_windows, window_size)
    power = np.mean(windows ** 2, axis=1)
    rms = np.sqrt(power)
    peak_rms = float(np.percentile(rms, 95))
    if peak_rms <= 0:
        return None

    threshold = peak_rms * 0.05
    noise_power = power[rms < threshold]
    speech_power = power[rms >= threshold]
    if noise_power.size == 0 or speech_power.size == 0:
        return None

    mean_noise_power = float(np.mean(noise_power))
    mean_speech_power = float(np.mean(speech_power))
    if mean_noise_power <= 0:
        return None

    return float(10.0 * np.log10(mean_speech_power / mean_noise_power))


def compute_clipping_ratio(audio_path: str, *, near_full_scale: float = 0.99) -> Optional[float]:
    """Trả về tỉ lệ sample WAV PCM 16-bit chạm hoặc gần biên độ tối đa."""
    try:
        with wave.open(audio_path, "rb") as wav_file:
            sample_width = wav_file.getsampwidth()
            raw = wav_file.readframes(wav_file.getnframes())
    except (wave.Error, OSError, EOFError):
        return None

    if not raw or sample_width != 2:
        return None

    samples = np.frombuffer(raw, dtype=np.int16)
    if samples.size == 0:
        return None

    clip_level = near_full_scale * 32767
    clipped = int(np.sum(np.abs(samples.astype(np.int32)) >= clip_level))
    return clipped / samples.size


def _fold_vietnamese_diacritics(text: str) -> str:
    """Bỏ dấu tiếng Việt chỉ để so khớp, không dùng cho transcript hiển thị."""
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_for_wer(text: str) -> str:
    """Chuẩn hóa hai transcript trước khi tính WER theo từ."""
    normalized = _fold_vietnamese_diacritics(unwrap_language_tags(text).lower())
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def word_error_rate(reference_words: List[str], hypothesis_words: List[str]) -> Optional[float]:
    """Tính WER bằng Levenshtein trên danh sách từ."""
    if not reference_words and not hypothesis_words:
        return None
    if not reference_words:
        return 1.0

    n, m = len(reference_words), len(hypothesis_words)
    distances = list(range(m + 1))
    for i in range(1, n + 1):
        previous_diagonal, distances[0] = distances[0], i
        for j in range(1, m + 1):
            previous_row = distances[j]
            distances[j] = (
                previous_diagonal
                if reference_words[i - 1] == hypothesis_words[j - 1]
                else 1 + min(previous_diagonal, distances[j], distances[j - 1])
            )
            previous_diagonal = previous_row
    return distances[m] / n


def compute_cross_asr_agreement(text_primary: str, text_audit: str) -> Optional[float]:
    """A = 1 - WER_norm giữa transcript chính và transcript audit."""
    reference = normalize_for_wer(text_primary).split()
    hypothesis = normalize_for_wer(text_audit).split()
    wer = word_error_rate(reference, hypothesis)
    return None if wer is None else max(0.0, 1.0 - wer)


def transcribe(audio_path: str, language: str) -> Optional[str]:
    """Transcribe the full audio file. See _transcribe_impl for behavior;
    this wrapper discards the per-segment confidence average for callers that
    only need the text (e.g. the live-exam archive path)."""
    text, _confidence = _transcribe_impl(audio_path, language)
    return text


def transcribe_with_confidence(audio_path: str, language: str) -> Tuple[Optional[str], Optional[float]]:
    """Same as transcribe(), but also returns the word-count-weighted average
    per-segment ASR confidence (0-1, or None if unavailable) -- used at eval
    time to populate asr_confidence_avg (see AnswerLengthNode), which feeds
    both the audioQuality signal and overallConfidence."""
    return _transcribe_impl(audio_path, language)


def _transcribe_impl(audio_path: str, language: str) -> Tuple[Optional[str], Optional[float]]:
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
            logger.warning(
                "[transcribe] recognized event with no usable text audio_path=%s reason=%s",
                audio_path, result.reason,
            )
            return
        detected = speechsdk.AutoDetectSourceLanguageResult(result).language or language
        segments.append(_TranscribedSegment(text=result.text, language=detected, confidence=_segment_confidence(result)))

    def on_canceled(evt) -> None:
        nonlocal canceled_error
        if evt.reason == speechsdk.CancellationReason.Error:
            canceled_error = evt.error_details
            logger.warning(
                "[transcribe] recognition canceled with error audio_path=%s error=%s",
                audio_path, evt.error_details,
            )
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
        finished = done.wait(timeout=timeout_seconds)
    finally:
        recognizer.stop_continuous_recognition()

    if not finished:
        logger.warning(
            "[transcribe] timed out waiting for recognition to finish audio_path=%s "
            "audio_duration=%s timeout=%s segments_so_far=%d",
            audio_path, audio_duration, timeout_seconds, len(segments),
        )

    if canceled_error:
        raise RuntimeError(f"Azure continuous recognition canceled: {canceled_error}")

    if not segments:
        logger.warning(
            "[transcribe] no speech segments recognized audio_path=%s audio_duration=%s",
            audio_path, audio_duration,
        )
        return None, None

    parts = [
        seg.text if seg.language.lower().startswith(target_prefix) else _wrap_non_target_segment(seg.text, seg.language)
        for seg in segments
    ]

    confidences = [seg.confidence for seg in segments if seg.confidence is not None]
    # Weighted by word count so a short segment's confidence doesn't count as much as a long
    # one -- mirrors merge_pronunciation_summaries' weighting in pronunciation_eval_node_config.
    weights = [max(len(seg.text.split()), 1) for seg in segments if seg.confidence is not None]
    avg_confidence = sum(c * w for c, w in zip(confidences, weights)) / sum(weights) if confidences else None

    return " ".join(parts), avg_confidence
