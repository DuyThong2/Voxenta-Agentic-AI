"""Legacy avatar TTS/WebRTC scaffolding kept for possible future hosted-avatar work.

These modules are not part of the live realtime flow today: nothing in the active exam path calls
this file, and the remaining avatar WebRTC path stays disabled on the WPF side unless
AppSettings.EnableAvatarWebRtc is turned back on. They remain here as a reference starting point
if a future hosted avatar (for example Azure realtime avatar synthesis) is wired in.

Ties TTS -> avatar WebRTC playback together for one utterance (Phase 4 of
docs/realtime-self-hosted-avatar-plan.md, audio-only mode).

Called by AttemptConnection whenever the avatar needs to say something: the question prompt at
question_start, and decision.next_prompt_text (or constants.CLOSING_REPLY) after each turn_end
decision. Runs as a fire-and-forget background task from the caller's perspective -- same spirit
as turn_publisher.publish_turn_if_new -- so speaking the next utterance never blocks the WebSocket
message loop or the decision response already sent to the client.

Video rendering (LivePortrait+MuseTalk, formerly realtime/avatar_renderer.py) was removed: it was
confirmed multiple-x slower than realtime on this hardware (see project memory). This now plays
the synthesized WAV directly through AvatarTrackPair -- aiortc's MediaPlayer reports no video
stream for an audio-only file, so AvatarVideoTrack's existing idle-frame fallback covers the video
slot automatically (no avatar_webrtc.py change needed). A phone-call-style placeholder until a
hosted avatar (e.g. Azure's realtime avatar) is wired in.

Utterances for one exam attempt are spoken strictly one at a time, in the order they were
scheduled (see _get_lock below) -- AttemptConnection schedules each speak() call as a
fire-and-forget asyncio task with no ordering guarantee of its own, so two calls close together
(a turn_end's closing remark immediately followed by the next question_start's prompt) would
otherwise race independently through TTS synthesis; whichever finishes first calls
track_pair.play() first, which can cut off or entirely skip the other utterance's audio.
"""

import asyncio
import logging
import wave
from pathlib import Path
from typing import Dict, Optional

from realtime._legacy_avatar import avatar_webrtc, tts_client

logger = logging.getLogger(__name__)

AGENTS_DIR = Path(__file__).resolve().parent.parent.parent
UTTERANCE_DIR = AGENTS_DIR / "var" / "avatar" / "utterances"

# Margin added to each utterance's measured WAV duration before letting the next queued speak()
# proceed AND before avatar_utterance_complete is sent (see attempt_connection.py) -- the WAV's
# own duration doesn't account for the network bridge (WebRTC jitter buffer, decode/playback on
# the WPF side) actually delivering that audio to the listener. Confirmed live: 0.3s was too
# tight under real network conditions -- WPF opened the student's mic (on receiving
# avatar_utterance_complete) before the student had actually finished hearing the question.
# There's no true "playback ended" signal from the client to wait on instead (WebRTC audio is a
# continuous stream with no utterance-boundary event), so this stays a conservative fixed margin
# -- mirrors the 4s buffer ExamViewModel.cs already uses for the same "make sure audio really
# finished" concern at exam completion.
_PLAYBACK_MARGIN_SECONDS = 1.5

# One lock per exam_attempt_id, never explicitly cleaned up -- each is a few bytes for the
# attempt's lifetime, which is bounded by the process anyway (a fresh attempt_id per exam attempt,
# never reused), so this isn't worth the complexity of wiring up a cleanup hook.
_speak_locks: Dict[str, asyncio.Lock] = {}


def _get_lock(exam_attempt_id: str) -> asyncio.Lock:
    lock = _speak_locks.get(exam_attempt_id)
    if lock is None:
        lock = asyncio.Lock()
        _speak_locks[exam_attempt_id] = lock
    return lock


def _wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wav_file:
        return wav_file.getnframes() / float(wav_file.getframerate())


async def speak(exam_attempt_id: str, text: str, *, sequence: int, rate: Optional[str] = None) -> None:
    if not text or not text.strip():
        return

    track_pair = avatar_webrtc.get_track_pair(exam_attempt_id)
    if track_pair is None:
        logger.info(
            "[avatar_speech] no avatar WebRTC connection for exam_attempt_id=%s, skipping speech",
            exam_attempt_id,
        )
        return

    # Holds the lock for this utterance's full synthesize+play+playback-duration so a
    # closely-following speak() call for the same attempt (already queued behind this lock, since
    # asyncio.Lock wakes waiters in FIFO order) can't preempt or talk over this one.
    async with _get_lock(exam_attempt_id):
        try:
            utterance_dir = UTTERANCE_DIR / exam_attempt_id
            utterance_dir.mkdir(parents=True, exist_ok=True)
            wav_path = utterance_dir / f"{sequence:04d}.wav"

            await tts_client.synthesize_to_wav_async(text, wav_path, rate=rate)
            await track_pair.play(wav_path)
            duration = await asyncio.to_thread(_wav_duration_seconds, wav_path)
            await asyncio.sleep(duration + _PLAYBACK_MARGIN_SECONDS)
        except Exception:
            logger.exception(
                "[avatar_speech] failed to synthesize/play utterance exam_attempt_id=%s sequence=%d",
                exam_attempt_id, sequence,
            )
