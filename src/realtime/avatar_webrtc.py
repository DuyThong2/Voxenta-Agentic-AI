"""Avatar WebRTC publisher (Phase 4 of docs/realtime-self-hosted-avatar-plan.md).

A SEPARATE aiortc RTCPeerConnection from controller/webrtc.py's proctoring connection (recvonly
camera->YOLO) -- this one sends the rendered avatar's video+audio TO the student. Opened once per
exam attempt and held open across every question (the plan's no-reconnect-gap design): the two
outbound tracks are added to the peer connection exactly once, at offer/answer time, and never
re-negotiated. Each question's/turn's rendered utterance video is swapped in via
AvatarTrackPair.play() instead of adding new tracks or renegotiating.
"""

import asyncio
import logging
import time
from fractions import Fraction
from pathlib import Path
from typing import Dict, Optional

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack, VideoStreamTrack
from av import AudioFrame, VideoFrame

logger = logging.getLogger(__name__)

_IDLE_AUDIO_SAMPLE_RATE = 48000
_IDLE_AUDIO_PTIME = 0.020


def _build_idle_video_frame() -> VideoFrame:
    frame = VideoFrame(width=640, height=480)
    for plane in frame.planes:
        plane.update(bytes(plane.buffer_size))
    return frame


class AvatarVideoTrack(VideoStreamTrack):
    """Outbound video track for one exam attempt's avatar connection. Delegates to whichever
    MediaPlayer sub-track is currently playing a rendered utterance; falls back to a blank idle
    frame between utterances. pts/time_base are always re-stamped from this track's own
    continuous next_timestamp() counter (inherited from VideoStreamTrack), never the source
    file's own pts -- so switching between many short-lived per-utterance files never produces a
    backward-jumping RTP timestamp at the receiver."""

    def __init__(self) -> None:
        super().__init__()
        self._source: Optional[MediaStreamTrack] = None

    def set_source(self, track: Optional[MediaStreamTrack]) -> None:
        self._source = track

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        frame = None
        if self._source is not None:
            try:
                frame = await self._source.recv()
            except MediaStreamError:
                self._source = None
        if frame is None:
            # A fresh frame each call, not one shared mutable instance -- aiortc's sender encodes
            # each recv()'d frame before fetching the next one, so reusing one instance happens to
            # be safe in that specific call pattern, but is a latent footgun (e.g. for tests or
            # any future buffering) if anything ever holds two outstanding frame references.
            frame = _build_idle_video_frame()
        frame.pts = pts
        frame.time_base = time_base
        return frame


class AvatarAudioTrack(MediaStreamTrack):
    """Outbound audio track, same delegate-or-idle-silence design as AvatarVideoTrack. Idle
    silence pacing restarts fresh each time delegation ends (see set_source) -- harmless since
    aiortc's Opus encoder derives RTP timestamps from its own internal sample counter, not from
    these frames' pts (confirmed directly in aiortc/codecs/opus.py: OpusEncoder.encode resamples
    every incoming frame through its own AudioResampler and computes the RTP timestamp from the
    resampler's own output packets, never reading the input frame's .pts)."""

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._source: Optional[MediaStreamTrack] = None
        self._idle_start: Optional[float] = None
        self._idle_timestamp = 0

    def set_source(self, track: Optional[MediaStreamTrack]) -> None:
        self._source = track
        if track is not None:
            self._idle_start = None

    async def recv(self) -> AudioFrame:
        if self._source is not None:
            try:
                return await self._source.recv()
            except MediaStreamError:
                self._source = None
                self._idle_start = None

        samples = int(_IDLE_AUDIO_PTIME * _IDLE_AUDIO_SAMPLE_RATE)
        now = time.time()
        if self._idle_start is None:
            self._idle_start = now
            self._idle_timestamp = 0
        else:
            self._idle_timestamp += samples
            wait = self._idle_start + (self._idle_timestamp / _IDLE_AUDIO_SAMPLE_RATE) - now
            if wait > 0:
                await asyncio.sleep(wait)

        frame = AudioFrame(format="s16", layout="mono", samples=samples)
        for plane in frame.planes:
            plane.update(bytes(plane.buffer_size))
        frame.pts = self._idle_timestamp
        frame.sample_rate = _IDLE_AUDIO_SAMPLE_RATE
        frame.time_base = Fraction(1, _IDLE_AUDIO_SAMPLE_RATE)
        return frame


class AvatarTrackPair:
    """One per exam attempt: the two outbound tracks plus whichever MediaPlayer is currently
    playing an utterance's rendered video into them."""

    def __init__(self) -> None:
        self.video = AvatarVideoTrack()
        self.audio = AvatarAudioTrack()
        self._player: Optional[MediaPlayer] = None
        self._lock = asyncio.Lock()

    async def play(self, video_path: Path) -> None:
        """Switch both tracks to a new rendered utterance file. One utterance plays at a time per
        attempt (the avatar never overlaps two utterances) -- the lock just makes that explicit
        if two play() calls ever race."""
        async with self._lock:
            old_player = self._player
            player = MediaPlayer(str(video_path))
            self._player = player
            self.video.set_source(player.video)
            self.audio.set_source(player.audio)
            self._stop_player(old_player)

    def stop(self) -> None:
        self.video.set_source(None)
        self.audio.set_source(None)
        self._stop_player(self._player)
        self._player = None

    @staticmethod
    def _stop_player(player: Optional[MediaPlayer]) -> None:
        if player is None:
            return
        if player.video is not None:
            player.video.stop()
        if player.audio is not None:
            player.audio.stop()


_pcs: Dict[str, RTCPeerConnection] = {}
_track_pairs: Dict[str, AvatarTrackPair] = {}


def get_track_pair(exam_attempt_id: str) -> Optional[AvatarTrackPair]:
    return _track_pairs.get(exam_attempt_id)


async def handle_offer(exam_attempt_id: str, sdp: str, type_: str) -> RTCSessionDescription:
    """One avatar connection per exam attempt: a new offer for an exam_attempt_id that already
    has a live connection (e.g. a reconnect) replaces it outright rather than stacking a second
    one."""
    await close_connection(exam_attempt_id)

    track_pair = AvatarTrackPair()
    _track_pairs[exam_attempt_id] = track_pair

    pc = RTCPeerConnection()
    _pcs[exam_attempt_id] = pc
    pc.addTrack(track_pair.video)
    pc.addTrack(track_pair.audio)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        logger.info("[avatar_webrtc] exam_attempt_id=%s state=%s", exam_attempt_id, pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await close_connection(exam_attempt_id)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type_))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    logger.info("[avatar_webrtc] connection opened exam_attempt_id=%s", exam_attempt_id)
    return pc.localDescription


async def close_connection(exam_attempt_id: str) -> None:
    pc = _pcs.pop(exam_attempt_id, None)
    if pc is not None:
        await pc.close()
    track_pair = _track_pairs.pop(exam_attempt_id, None)
    if track_pair is not None:
        track_pair.stop()


async def close_all_connections() -> None:
    logger.info("[avatar_webrtc] closing %d active connections...", len(_pcs))
    for exam_attempt_id in list(_pcs.keys()):
        await close_connection(exam_attempt_id)
