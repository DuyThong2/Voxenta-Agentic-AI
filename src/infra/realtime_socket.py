"""Thin network-client wrapper around FastAPI's WebSocket for the realtime
exam attempt connection. Pure relay: sends whatever payload it's given,
classifies/parses incoming frames, and raises WebSocketDisconnect on
disconnect -- no business decisions (what to send, how to react to a frame)
live here; those stay in realtime.attempt_connection.AttemptConnection.
"""

import json
import asyncio
import logging
from typing import Any, AsyncIterator, Tuple

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class RealtimeSocket:
    """Owns exactly one FastAPI WebSocket for the lifetime of one exam
    attempt connection."""

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        # Tuan tu hoa moi lan gui: socket nay co NHIEU nguoi ghi chay song song -- vong xu ly
        # message chinh, cac task nen do _speak sinh ra qua spawn(), va nhip tim. Gui chong nhau
        # tren cung mot WebSocket cua Starlette la khong an toan, ma send_json ben duoi lai nuot
        # moi ngoai le nen loi kieu do se bien mat lang le, keo theo mat mot ack hoac mot decision.
        self._send_lock = asyncio.Lock()

    async def accept(self) -> None:
        await self._websocket.accept()

    async def close(self, code: int = 1000) -> None:
        await self._websocket.close(code=code)

    async def send_json(self, payload: dict) -> None:
        """Never raises -- a failed send is logged and swallowed, same as
        every call site in AttemptConnection already did individually before
        this wrapper existed (and the ones that didn't were only reachable
        from realtime_controller.py's own top-level try/except around message
        handling, which already just logs and continues -- so this is the
        same effective behavior, just consolidated in one place)."""
        try:
            async with self._send_lock:
                await self._websocket.send_json(payload)
        except Exception:
            logger.exception(
                "[realtime_socket] failed to send payload type=%s",
                payload.get("type"),
            )

    async def iter_frames(self) -> AsyncIterator[Tuple[str, Any]]:
        """Yields ("audio", bytes) for a binary frame or ("text", dict) for a
        parsed JSON text frame. Raises WebSocketDisconnect when the client
        disconnects. A text frame that isn't valid JSON is logged and
        skipped, not raised."""
        while True:
            message = await self._websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000), message.get("reason"))

            if message.get("bytes") is not None:
                yield ("audio", message["bytes"])
                continue

            text = message.get("text")
            if text is None:
                continue

            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                logger.warning("[realtime_socket] failed to parse text frame as JSON: %r", text)
                continue

            yield ("text", parsed)
