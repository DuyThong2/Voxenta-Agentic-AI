"""
Proctoring alert decision policy: when is a YOLO detection a real, reportable
violation vs. single-frame detector noise, and how often should the same
violation type be allowed to re-alert while it keeps holding.

Two gates, applied in order by the caller (controller/proctoring_frame_processor.py):
1. condition_confirmed -- hysteresis: the condition must hold for ALERT_STREAK_FRAMES
   consecutive processed frames before it's treated as real.
2. should_emit_alert -- edge-trigger + cooldown: fires immediately the first time,
   then at most once per ALERT_COOLDOWN_SECONDS while the condition keeps holding.
"""

import asyncio
import logging
import time
from collections import defaultdict
from typing import Dict, Set

from config.webrtc_config import settings
from infra.alert_client import push_alert

logger = logging.getLogger(__name__)

ALERT_COOLDOWN_SECONDS = settings.PROCTORING_ALERT_COOLDOWN_SECONDS
ALERT_STREAK_FRAMES = settings.PROCTORING_ALERT_STREAK_FRAMES

# session_id -> set of alert types currently considered "active" (condition still holds)
_active_alert_types: Dict[str, Set[str]] = defaultdict(set)
# session_id -> {alert_type: monotonic time it was last emitted}
_alert_last_emitted_at: Dict[str, Dict[str, float]] = defaultdict(dict)
# session_id -> {alert_type: consecutive processed-frame count the condition has held}
_condition_streak: Dict[str, Dict[str, int]] = defaultdict(dict)

_ALERT_TYPE_MAP = {
    "PERSON_MISSING": "PERSON_MISSING",
    "MULTIPLE_PERSONS": "MULTIPLE_PERSONS",
}


def map_alert_type(event: dict) -> str:
    if event.get("type") == "OBJECT_DETECTED":
        return "PHONE_DETECTED" if event.get("object") == "cell phone" else "OBJECT_DETECTED"
    return _ALERT_TYPE_MAP.get(str(event.get("type") or ""), str(event.get("type") or ""))


def should_emit_alert(session_id: str, alert_type: str) -> bool:
    """Edge-triggered + cooldown gate: fires immediately the first time a condition
    appears, then at most once per ALERT_COOLDOWN_SECONDS while it keeps holding."""
    now = time.monotonic()
    active = _active_alert_types[session_id]
    last_emitted = _alert_last_emitted_at[session_id]

    is_new = alert_type not in active
    cooldown_elapsed = alert_type not in last_emitted or (now - last_emitted[alert_type]) >= ALERT_COOLDOWN_SECONDS

    if not (is_new or cooldown_elapsed):
        return False

    active.add(alert_type)
    last_emitted[alert_type] = now
    return True


def clear_alert(session_id: str, alert_type: str) -> None:
    """Condition no longer holds -- next occurrence is treated as a fresh edge, not
    a continuation still waiting out the old cooldown."""
    _active_alert_types[session_id].discard(alert_type)


def condition_confirmed(session_id: str, alert_type: str) -> bool:
    """Hysteresis: increments the consecutive-frame streak for this condition and returns
    True only once it has held for ALERT_STREAK_FRAMES processed frames in a row. Call
    reset_streak instead the moment the condition stops holding."""
    streaks = _condition_streak[session_id]
    streaks[alert_type] = streaks.get(alert_type, 0) + 1
    return streaks[alert_type] >= ALERT_STREAK_FRAMES


def reset_streak(session_id: str, alert_type: str) -> None:
    _condition_streak[session_id][alert_type] = 0


def schedule_alert(session_id: str, event: dict) -> None:
    alert_type = map_alert_type(event)
    if not alert_type:
        return

    try:
        asyncio.get_running_loop().create_task(
            push_alert(
                room_id=session_id,
                participant_id=session_id,
                stream_id=session_id,
                alert_type=alert_type,
                confidence=float(event.get("confidence") or 1.0),
            )
        )
    except RuntimeError:
        logger.warning("[PROCTORING] no running loop, skipping alert for session=%s", session_id)


def clear_session(session_id: str) -> None:
    """Drop all per-session policy state -- called by webrtc_session.cleanup_session
    once a session ends, so a later reused session_id starts with a clean slate."""
    _active_alert_types.pop(session_id, None)
    _alert_last_emitted_at.pop(session_id, None)
    _condition_streak.pop(session_id, None)
