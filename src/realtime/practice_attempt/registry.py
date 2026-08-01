"""Process-local registry for active practice realtime connections -- parallel to
realtime/attempt/registry.py, keyed by practice_session_id instead of exam_attempt_id."""

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from realtime.practice_attempt.connection import PracticeAttemptConnection

logger = logging.getLogger(__name__)

_active_connections: dict[str, "PracticeAttemptConnection"] = {}


def register_practice_attempt_connection(connection: "PracticeAttemptConnection") -> None:
    _active_connections[connection.practice_session_id] = connection


def unregister_practice_attempt_connection(connection: "PracticeAttemptConnection") -> None:
    existing = _active_connections.get(connection.practice_session_id)
    if existing is connection:
        _active_connections.pop(connection.practice_session_id, None)


def get_practice_attempt_connection(practice_session_id: str) -> Optional["PracticeAttemptConnection"]:
    return _active_connections.get(practice_session_id)


async def close_all_practice_attempt_connections() -> None:
    for connection in list(_active_connections.values()):
        try:
            await connection.close()
        except Exception:
            logger.exception(
                "[practice_attempt_registry] failed to close practice_session_id=%s",
                connection.practice_session_id,
            )
    _active_connections.clear()
