"""Production Azure Voice Live (realtime STT+VAD) client (Phase 3 of
docs/realtime-self-hosted-avatar-plan.md).

Promotes spikes/voice_live_poc.py's connection/event-translation logic to a
long-lived client: one VoiceLiveClient is opened per exam attempt
(AttemptConnection owns it) and stays connected for every question in the
attempt, instead of the PoC's single-WAV-then-exit run. Voice Live's own
per-utterance conversation items naturally segment multiple questions'
speech without needing to reconnect between them.

Deliberately STT+VAD only: turn_detection.create_response=False, same as the
PoC -- the existing LangGraph follow-up decision logic
(node/followUpDecisionGraph) is what decides what the avatar says next, not
Voice Live's own model.
"""

import asyncio
import base64
import logging
import os
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

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

logger = logging.getLogger(__name__)

# Matches TurnAudioRecorder.cs (16kHz, 16-bit, mono) -- WPF's MicAudioStreamer (Phase 5) streams
# this format directly, no resampling needed.
EXPECTED_SAMPLE_RATE = 16_000


@dataclass
class VoiceLiveServerEvent:
    kind: str  # "vad_speech_start" | "vad_speech_end" | "partial_transcript" | "final_transcript" | "error"
    text: Optional[str] = None


def _load_config() -> dict:
    api_key = os.getenv("AZURE_VOICELIVE_API_KEY")
    endpoint = os.getenv("AZURE_VOICELIVE_ENDPOINT")
    if not api_key or not endpoint:
        raise RuntimeError(
            "AZURE_VOICELIVE_API_KEY/AZURE_VOICELIVE_ENDPOINT must be set (see agents/.env, "
            "agents/spikes/README.md section 1)."
        )
    return {
        "api_key": api_key,
        "endpoint": endpoint,
        "model": os.getenv("AZURE_VOICELIVE_MODEL", "gpt-realtime-mini"),
        "api_version": os.getenv("AZURE_VOICELIVE_API_VERSION", "2026-04-10"),
        # gpt-realtime / gpt-realtime-mini support whisper-1, gpt-4o-transcribe,
        # gpt-4o-mini-transcribe, gpt-4o-transcribe-diarize (NOT azure-speech, which is for
        # non-multimodal models only) -- see spikes/voice_live_poc.py for how this was confirmed.
        "transcription_model": os.getenv("AZURE_VOICELIVE_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe"),
    }


class VoiceLiveClient:
    """One instance per exam attempt, owned by AttemptConnection. `on_event`
    is called for every translated server event; the caller (AttemptConnection)
    is responsible for routing it to the currently-active RealtimeExamSession
    and/or forwarding it to the WPF client."""

    def __init__(self, on_event: Callable[[VoiceLiveServerEvent], Awaitable[None]]) -> None:
        self._on_event = on_event
        self._config = _load_config()
        self._conn = None
        self._cm = None
        self._consume_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._cm = connect(
            endpoint=self._config["endpoint"],
            credential=AzureKeyCredential(self._config["api_key"]),
            model=self._config["model"],
            api_version=self._config["api_version"],
        )
        self._conn = await self._cm.__aenter__()
        await self._conn.session.update(
            session=RequestSession(
                modalities=[Modality.TEXT, Modality.AUDIO],
                input_audio_format=InputAudioFormat.PCM16,
                input_audio_sampling_rate=EXPECTED_SAMPLE_RATE,
                input_audio_transcription=AudioInputTranscriptionOptions(
                    model=self._config["transcription_model"],
                ),
                turn_detection=ServerVad(
                    threshold=0.5,
                    prefix_padding_ms=300,
                    silence_duration_ms=500,
                    create_response=False,
                ),
            )
        )
        self._consume_task = asyncio.create_task(self._consume_loop())
        logger.info("[voice_live_client] connected (model=%s)", self._config["model"])

    async def push_audio(self, pcm16_bytes: bytes) -> None:
        """Forward one chunk of raw PCM16 mic audio. Safe to call with
        whatever chunk size WPF's MicAudioStreamer picks -- Voice Live does
        not require a fixed frame size per append call."""
        if self._conn is None or not pcm16_bytes:
            return
        await self._conn.input_audio_buffer.append(audio=base64.b64encode(pcm16_bytes).decode("ascii"))

    async def _consume_loop(self) -> None:
        try:
            async for event in self._conn:
                translated = self._translate(event)
                if translated is not None:
                    await self._on_event(translated)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[voice_live_client] consume loop failed")
            await self._on_event(VoiceLiveServerEvent(kind="error", text="voice_live_consume_failed"))

    @staticmethod
    def _translate(event) -> Optional[VoiceLiveServerEvent]:
        if event.type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
            return VoiceLiveServerEvent(kind="vad_speech_start")
        if event.type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED:
            return VoiceLiveServerEvent(kind="vad_speech_end")
        if event.type == ServerEventType.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_DELTA:
            return VoiceLiveServerEvent(kind="partial_transcript", text=event.delta)
        if event.type == ServerEventType.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED:
            return VoiceLiveServerEvent(kind="final_transcript", text=event.transcript)
        if event.type == ServerEventType.ERROR:
            return VoiceLiveServerEvent(kind="error", text=str(event))
        return None

    async def close(self) -> None:
        if self._consume_task is not None:
            self._consume_task.cancel()
            try:
                await self._consume_task
            except (asyncio.CancelledError, Exception):
                pass
            self._consume_task = None
        if self._cm is not None:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception:
                logger.exception("[voice_live_client] error closing connection")
            self._cm = None
        self._conn = None
