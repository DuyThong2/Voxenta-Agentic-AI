"""Process-local registry for active realtime attempt connections."""

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from realtime.attempt.connection import AttemptConnection

logger = logging.getLogger(__name__)

_active_connections: dict[str, "AttemptConnection"] = {}


def register_attempt_connection(connection: "AttemptConnection") -> None:
    _active_connections[connection.exam_attempt_id] = connection


def unregister_attempt_connection(connection: "AttemptConnection") -> None:
    existing = _active_connections.get(connection.exam_attempt_id)
    if existing is connection:
        _active_connections.pop(connection.exam_attempt_id, None)


def get_attempt_connection(exam_attempt_id: str) -> Optional["AttemptConnection"]:
    return _active_connections.get(exam_attempt_id)


async def close_all_attempt_connections() -> None:
    for connection in list(_active_connections.values()):
        try:
            await connection.close()
        except Exception:
            logger.exception(
                "[attempt_registry] failed to close exam_attempt_id=%s",
                connection.exam_attempt_id,
            )
    _active_connections.clear()
