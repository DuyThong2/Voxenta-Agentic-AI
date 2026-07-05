"""Avatar WebRTC offer/answer endpoint (Phase 4 of docs/realtime-self-hosted-avatar-plan.md) --
parallel in shape to controller/webrtc.py's /webrtc/offer, but reversed media direction (the
server publishes avatar video+audio to the student) and scoped to exam_attempt_id (one
connection per exam attempt) instead of a freshly-minted session_id.
"""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from realtime import avatar_webrtc

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/avatar/webrtc", tags=["AvatarWebRTC"])


@router.post("/offer/{exam_attempt_id}")
async def offer(exam_attempt_id: str, request: Request):
    params = await request.json()
    if "sdp" not in params or "type" not in params:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid WebRTC offer. Body must include 'sdp' and 'type'."},
        )

    answer = await avatar_webrtc.handle_offer(exam_attempt_id, params["sdp"], params["type"])
    return JSONResponse({"sdp": answer.sdp, "type": answer.type})


@router.delete("/{exam_attempt_id}")
async def disconnect(exam_attempt_id: str):
    await avatar_webrtc.close_connection(exam_attempt_id)
    return {"status": "disconnected", "exam_attempt_id": exam_attempt_id}
