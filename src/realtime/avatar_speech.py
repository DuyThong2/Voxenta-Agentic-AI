"""Ties TTS -> avatar rendering -> avatar WebRTC playback together for one utterance (Phase 4 of
docs/realtime-self-hosted-avatar-plan.md).

Called by AttemptConnection whenever the avatar needs to say something: the question prompt at
question_start, and decision.next_prompt_text (or constants.CLOSING_REPLY) after each turn_end
decision. Runs as a fire-and-forget background task from the caller's perspective -- same spirit
as turn_publisher.publish_turn_if_new -- so rendering/speaking the next utterance never blocks the
WebSocket message loop or the decision response already sent to the client.
"""

import logging

from realtime import avatar_renderer, avatar_webrtc, gpu_scheduler, tts_client

logger = logging.getLogger(__name__)

UTTERANCE_DIR = avatar_renderer.AVATAR_VAR_DIR / "utterances"


async def speak(exam_attempt_id: str, text: str, *, sequence: int) -> None:
    if not text or not text.strip():
        return

    track_pair = avatar_webrtc.get_track_pair(exam_attempt_id)
    if track_pair is None:
        logger.info(
            "[avatar_speech] no avatar WebRTC connection for exam_attempt_id=%s, skipping render",
            exam_attempt_id,
        )
        return

    try:
        utterance_dir = UTTERANCE_DIR / exam_attempt_id
        utterance_dir.mkdir(parents=True, exist_ok=True)
        wav_path = utterance_dir / f"{sequence:04d}.wav"
        video_path = utterance_dir / f"{sequence:04d}.mp4"

        async with gpu_scheduler.gpu_lock():
            # ensure_avatar_ready is idempotent (checks its own on-disk cache first) -- calling it
            # defensively before every utterance, rather than tracking "did I already set this
            # attempt up" in memory, matches this pipeline's durable-idempotency convention (see
            # turn_publisher.py).
            await avatar_renderer.ensure_avatar_ready(exam_attempt_id)
            await tts_client.synthesize_to_wav_async(text, wav_path)
            await avatar_renderer.render_utterance(exam_attempt_id, wav_path, video_path)

        await track_pair.play(video_path)
    except Exception:
        logger.exception(
            "[avatar_speech] failed to render/play utterance exam_attempt_id=%s sequence=%d",
            exam_attempt_id, sequence,
        )
