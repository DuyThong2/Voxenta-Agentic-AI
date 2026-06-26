"""Azure TTS streaming synthesis (Phase 4 of docs/realtime-self-hosted-avatar-plan.md).

Reuses the same Azure Cognitive Services Speech resource (AZURE_SPEECH_KEY/AZURE_SPEECH_REGION)
already used for archive_graph's STT (utils/speech_client.py) -- one Speech resource handles both
directions. Synthesizes decision.next_prompt_text (or constants.CLOSING_REPLY) to a WAV file on
disk, since that's what avatar_renderer.py's MuseTalk stage needs as input (an audio file path,
not a raw byte stream).
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

import azure.cognitiveservices.speech as speechsdk

from utils import load_root_dotenv

logger = logging.getLogger(__name__)

# Matches the 16kHz/16-bit/mono convention used everywhere else in this pipeline
# (TurnAudioRecorder.cs, Voice Live's input format) -- avoids a resample step before MuseTalk's
# librosa.load(..., sr=16000) call in avatar_renderer.py.
_OUTPUT_FORMAT = speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm


def _build_synthesizer(voice_name: Optional[str]) -> speechsdk.SpeechSynthesizer:
    load_root_dotenv()
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION")
    if not speech_key or not speech_region:
        raise RuntimeError("Missing AZURE_SPEECH_KEY/AZURE_SPEECH_REGION in environment variables")

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.set_speech_synthesis_output_format(_OUTPUT_FORMAT)
    speech_config.speech_synthesis_voice_name = voice_name or os.getenv("AZURE_TTS_VOICE", "en-US-JennyNeural")
    return speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)


def synthesize_to_wav(text: str, out_wav_path: Path, *, voice_name: Optional[str] = None) -> Path:
    """Synchronous (the SDK's *_async methods still block the calling thread on .get()) -- call
    via synthesize_to_wav_async from async code, same to_thread pattern as avatar_renderer.py's
    torch inference."""
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    synthesizer = _build_synthesizer(voice_name)
    result = synthesizer.speak_text_async(text).get()

    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        cancellation = result.cancellation_details
        raise RuntimeError(
            f"Azure TTS synthesis failed: reason={result.reason} "
            f"details={cancellation.error_details if cancellation else None}"
        )

    out_wav_path.write_bytes(result.audio_data)
    logger.info("[tts_client] synthesized %d bytes -> %s", len(result.audio_data), out_wav_path)
    return out_wav_path


async def synthesize_to_wav_async(text: str, out_wav_path: Path, *, voice_name: Optional[str] = None) -> Path:
    return await asyncio.to_thread(synthesize_to_wav, text, out_wav_path, voice_name=voice_name)
