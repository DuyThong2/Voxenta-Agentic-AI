"""Phase 0 spike: drive Azure Voice Live (realtime STT+VAD) from a canned WAV file.

Standalone script, not wired into src/app.py — see spikes/README.md for setup. Run from the
`agents` directory:

    python spikes/voice_live_poc.py --wav data/<some_sample>.wav

Confirmed against the installed `azure-ai-voicelive==1.2.0` package source directly (not just
docs) — see .venv/Lib/site-packages/azure/ai/voicelive/. We deliberately set
`turn_detection.create_response=False`: this pipeline only wants Voice Live's STT+VAD, not its
own response generation — the existing LangGraph follow-up decision logic
(node/followUpDecisionGraph) is what decides what the avatar says next, not Voice Live's model.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import os
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

from azure.ai.voicelive.aio import connect
from azure.ai.voicelive.models import (
    AudioInputTranscriptionOptions,
    InputAudioFormat,
    Modality,
    RequestSession,
    ServerEventType,
    ServerVad,
)
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("voice_live_poc")

SPIKES_DIR = Path(__file__).resolve().parent
AGENTS_DIR = SPIKES_DIR.parent  # .env lives at agents/.env, not one level above that
RESULTS_FILE = SPIKES_DIR / "poc-results.md"

# Matches TurnAudioRecorder.cs (16kHz, 16-bit, mono) so a real captured turn can be fed in
# without any conversion. Voice Live's input_audio_sampling_rate only accepts 16000 or 24000.
EXPECTED_SAMPLE_RATE = 16_000
EXPECTED_SAMPLE_WIDTH_BYTES = 2
EXPECTED_CHANNELS = 1
CHUNK_MS = 20  # matches the 20-40ms frame size the plan's WPF mic-streaming design expects

# How much trailing silence to feed after real audio ends, so server-side VAD has something
# to actually detect silence in (see produce() below), and how long to keep listening after
# that for the async final-transcription event (which arrives after VAD speech_stopped).
TRAILING_SILENCE_MS = 1000
TAIL_WAIT_SECONDS = 3.0


@dataclass
class VoiceLiveEvent:
    kind: str  # "vad_speech_start" | "vad_speech_end" | "partial_transcript" | "final_transcript" | "error"
    text: Optional[str]
    elapsed_seconds: float = 0.0


@dataclass
class VoiceLiveSessionConfig:
    api_key: str
    endpoint: str
    model: str
    api_version: str
    transcription_model: str


def load_config() -> VoiceLiveSessionConfig:
    load_dotenv(dotenv_path=AGENTS_DIR / ".env")

    api_key = os.getenv("AZURE_VOICELIVE_API_KEY")
    endpoint = os.getenv("AZURE_VOICELIVE_ENDPOINT")
    model = os.getenv("AZURE_VOICELIVE_MODEL", "gpt-realtime-mini")
    api_version = os.getenv("AZURE_VOICELIVE_API_VERSION", "2026-04-10")
    # gpt-realtime / gpt-realtime-mini support whisper-1, gpt-4o-transcribe,
    # gpt-4o-mini-transcribe, gpt-4o-transcribe-diarize as the input transcription model
    # (NOT azure-speech, which is for non-multimodal models/agents only).
    transcription_model = os.getenv("AZURE_VOICELIVE_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe")

    missing = [
        name
        for name, value in (("AZURE_VOICELIVE_API_KEY", api_key), ("AZURE_VOICELIVE_ENDPOINT", endpoint))
        if not value
    ]
    if missing:
        raise SystemExit(
            "Missing required env var(s): "
            + ", ".join(missing)
            + ". See agents/spikes/README.md section 1 for how to set these up."
        )

    return VoiceLiveSessionConfig(
        api_key=api_key,  # type: ignore[arg-type]
        endpoint=endpoint,  # type: ignore[arg-type]
        model=model,
        api_version=api_version,
        transcription_model=transcription_model,
    )


def iter_pcm_chunks(wav_path: Path, chunk_ms: int = CHUNK_MS) -> Iterator[bytes]:
    """Yield raw PCM16 chunks from a WAV file, validating it matches the expected
    16kHz/16-bit/mono format TurnAudioRecorder.cs produces."""
    with wave.open(str(wav_path), "rb") as wav_file:
        if wav_file.getframerate() != EXPECTED_SAMPLE_RATE:
            raise ValueError(f"{wav_path} is {wav_file.getframerate()}Hz, expected {EXPECTED_SAMPLE_RATE}Hz")
        if wav_file.getsampwidth() != EXPECTED_SAMPLE_WIDTH_BYTES:
            raise ValueError(f"{wav_path} has {wav_file.getsampwidth() * 8}-bit samples, expected 16-bit")
        if wav_file.getnchannels() != EXPECTED_CHANNELS:
            raise ValueError(f"{wav_path} has {wav_file.getnchannels()} channels, expected mono")

        frames_per_chunk = int(EXPECTED_SAMPLE_RATE * (chunk_ms / 1000))
        while True:
            frames = wav_file.readframes(frames_per_chunk)
            if not frames:
                break
            yield frames


async def run_poc(wav_path: Path) -> List[VoiceLiveEvent]:
    config = load_config()
    events: List[VoiceLiveEvent] = []
    start = time.monotonic()
    stop_listening = asyncio.Event()

    def record(kind: str, text: Optional[str]) -> None:
        evt = VoiceLiveEvent(kind=kind, text=text, elapsed_seconds=time.monotonic() - start)
        logger.info("[%6.3fs] %s: %r", evt.elapsed_seconds, evt.kind, evt.text)
        events.append(evt)

    logger.info("Connecting to Azure Voice Live (model=%s, api_version=%s)...", config.model, config.api_version)

    async with connect(
        endpoint=config.endpoint,
        credential=AzureKeyCredential(config.api_key),
        model=config.model,
        api_version=config.api_version,
    ) as conn:
        await conn.session.update(
            session=RequestSession(
                modalities=[Modality.TEXT, Modality.AUDIO],
                input_audio_format=InputAudioFormat.PCM16,
                input_audio_sampling_rate=EXPECTED_SAMPLE_RATE,
                input_audio_transcription=AudioInputTranscriptionOptions(
                    model=config.transcription_model,
                ),
                turn_detection=ServerVad(
                    threshold=0.5,
                    prefix_padding_ms=300,
                    silence_duration_ms=500,
                    # We only want STT+VAD events here, not Voice Live's own conversational
                    # response — our LangGraph follow-up decision node decides what happens next.
                    create_response=False,
                ),
            )
        )

        bytes_per_chunk = int(EXPECTED_SAMPLE_RATE * (CHUNK_MS / 1000)) * EXPECTED_SAMPLE_WIDTH_BYTES
        silence_chunk = b"\x00" * bytes_per_chunk

        async def produce() -> None:
            for chunk in iter_pcm_chunks(wav_path):
                await conn.input_audio_buffer.append(audio=base64.b64encode(chunk).decode("ascii"))
                # Pace sends at real-time speed — VAD's silence_duration_ms is measured against
                # the audio's own timeline, so blasting all chunks instantly would make VAD
                # latency measurements meaningless.
                await asyncio.sleep(CHUNK_MS / 1000)

            # A WAV file has no trailing silence once it ends — server-side VAD needs actual
            # silent frames in the stream to detect speech_stopped; it can't infer silence just
            # because we stop sending. A live mic naturally provides this ambient silence after
            # the speaker stops talking, so simulate it here long enough to clear
            # turn_detection.silence_duration_ms (500ms) with margin.
            for _ in range(TRAILING_SILENCE_MS // CHUNK_MS):
                await conn.input_audio_buffer.append(audio=base64.b64encode(silence_chunk).decode("ascii"))
                await asyncio.sleep(CHUNK_MS / 1000)

            logger.info(
                "Finished streaming %s (+%dms trailing silence), waiting up to %.1fs for trailing events...",
                wav_path, TRAILING_SILENCE_MS, TAIL_WAIT_SECONDS,
            )
            # consume() may already set stop_listening early (e.g. once a final_transcript for
            # this single-utterance WAV has arrived) — don't sit through the full timeout if so.
            try:
                await asyncio.wait_for(stop_listening.wait(), timeout=TAIL_WAIT_SECONDS)
            except asyncio.TimeoutError:
                pass
            stop_listening.set()

        async def consume() -> None:
            async for event in conn:
                if event.type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
                    record("vad_speech_start", None)
                elif event.type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED:
                    record("vad_speech_end", None)
                elif event.type == ServerEventType.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_DELTA:
                    record("partial_transcript", event.delta)
                elif event.type == ServerEventType.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED:
                    record("final_transcript", event.transcript)
                    # This PoC streams a single WAV = a single utterance — once its final
                    # transcript has arrived there's nothing more useful to wait for, so stop
                    # immediately rather than sitting through the rest of TAIL_WAIT_SECONDS.
                    stop_listening.set()
                elif event.type == ServerEventType.ERROR:
                    record("error", str(event))
                    stop_listening.set()
                if stop_listening.is_set():
                    break

        await asyncio.gather(produce(), consume())

    return events


def append_results(wav_path: Path, events: List[VoiceLiveEvent]) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n## voice_live_poc — {wav_path.name}\n\n")
        if not events:
            f.write("(no events received)\n")
        for event in events:
            f.write(f"- `{event.elapsed_seconds:6.3f}s` {event.kind}: {event.text!r}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, required=True, help="Path to a 16kHz/16-bit/mono WAV file")
    args = parser.parse_args()

    try:
        events = asyncio.run(run_poc(args.wav))
    except KeyboardInterrupt:
        logger.warning("Interrupted by user — no results appended.")
        return
    append_results(args.wav, events)


if __name__ == "__main__":
    main()
