"""
Proctoring WebRTC session/connection bookkeeping + event storage/broadcast.

Owns the peer-connection registry and the per-session event log/SSE subscriber
set that controller/webrtc.py's routes read from. No business decisions here --
just tracking what exists and relaying events already decided elsewhere. Takes
no dependency on controller/proctoring_alert_policy.py (or anything else
business-decision-shaped) -- cleanup_session's optional on_cleanup callback
lets the caller (controller/webrtc.py) hook in any extra per-session teardown
without this module needing to know what that is.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Set

from aiortc import RTCPeerConnection

logger = logging.getLogger(__name__)

# --------------- State ---------------
pcs: Set[RTCPeerConnection] = set()
# session_id -> RTCPeerConnection
session_map: Dict[str, RTCPeerConnection] = {}
# session_id -> list of events
session_events: Dict[str, List[dict]] = defaultdict(list)
# session_id -> set of SSE subscribers (asyncio.Queue)
session_subscribers: Dict[str, Set[asyncio.Queue]] = defaultdict(set)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_event(event_type: str, message: str, confidence: float = 0.9, **extra):
    return {
        "type": event_type,
        "confidence": confidence,
        "message": message,
        "timestamp": utc_now(),
        **extra,
    }


async def broadcast_event(session_id: str, event: dict):
    """Push event to all SSE subscribers of a session."""
    session_events[session_id].append(event)
    for queue in session_subscribers.get(session_id, set()):
        await queue.put(event)


async def cleanup_session(session_id: str, *, on_cleanup: Optional[Callable[[str], None]] = None):
    """Clean up a session's peer connection and resources.

    on_cleanup, if given, runs after this module's own state is cleared --
    e.g. controller/webrtc.py passes proctoring_alert_policy.clear_session so
    that policy state gets reset too, without this module importing/knowing
    about the policy module at all.
    """
    pc = session_map.pop(session_id, None)
    if pc:
        await pc.close()
        pcs.discard(pc)

    # Notify SSE subscribers that session ended
    for queue in session_subscribers.get(session_id, set()):
        await queue.put({"type": "SESSION_ENDED", "timestamp": utc_now()})
    session_subscribers.pop(session_id, None)

    # Without this, a session_id that reconnects after a real end (same
    # exam_attempt_id, e.g. retrying the same exam) replays this session's
    # ENTIRE event history the instant the new SSE stream opens (see
    # controller/webrtc.py's stream_session_events, "Send existing events
    # first") -- confirmed live: old proctoring warnings from a prior,
    # already-ended attempt resurfaced immediately on a fresh connect,
    # unrelated to the client's actual current camera state. A reconnect
    # that happens WITHOUT cleanup_session having run yet (e.g. a network
    # blip mid-session) still replays correctly, since this only clears once
    # the session is genuinely over.
    session_events.pop(session_id, None)

    if on_cleanup is not None:
        on_cleanup(session_id)

    logger.info("[WEBRTC] Session %s cleaned up. Active: %d", session_id, len(pcs))


async def close_all_connections(*, on_cleanup: Optional[Callable[[str], None]] = None):
    """Called on app shutdown to clean up all peer connections."""
    logger.info("[WEBRTC] Closing %d active connections...", len(pcs))
    for session_id in list(session_map.keys()):
        await cleanup_session(session_id, on_cleanup=on_cleanup)
    logger.info("[WEBRTC] All connections closed.")
