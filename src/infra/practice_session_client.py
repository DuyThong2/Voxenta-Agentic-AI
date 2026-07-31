"""HTTP client Python -> Java for the practice realtime pivot (gói 11 mục 2.4/2.5).

Server-to-server only -- never depends on the student's own network connection, unlike
WPF's separate /turns/archive upload for exam (see task/implement/11-toi-uu-dung-de.md,
"Turn-recording trong phiên realtime" for why that pattern was rejected for practice).
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

_JAVA_BASE_URL = os.environ.get("VOX_JAVA_BASE_URL", "http://localhost:8080")
_INTERNAL_SECRET = os.environ.get("PRACTICE_INTERNAL_SECRET", "")
_NEXT_QUESTION_TIMEOUT_SECONDS = 8.0
_SUBMIT_TURN_TIMEOUT_SECONDS = 8.0
_UPLOAD_URL_TIMEOUT_SECONDS = 5.0
_S3_PUT_TIMEOUT_SECONDS = 15.0


def _headers() -> dict:
    return {"X-Internal-Secret": _INTERNAL_SECRET, "Content-Type": "application/json"}


async def request_next_question(practice_session_id: str) -> dict:
    """POST /internal/practice-sessions/{id}/next-question -- returns Java's
    {"message":..., "data": {"status": "ok"|"no_more_questions", "reason": ..., "question": {...}}}.
    Raises httpx.HTTPStatusError/TimeoutException on failure -- caller decides how to treat that
    (see PracticeQuestionSessionCoordinator.resolve_and_push_next_question)."""
    url = f"{_JAVA_BASE_URL}/internal/practice-sessions/{practice_session_id}/next-question"
    async with httpx.AsyncClient(timeout=_NEXT_QUESTION_TIMEOUT_SECONDS) as client:
        response = await client.post(url, headers=_headers())
        response.raise_for_status()
        return response.json()["data"]


async def submit_turn(practice_session_id: str, turn: dict) -> dict:
    """POST /internal/practice-sessions/{id}/turns -- turn matches
    PracticeSessionInternalController.TurnRequest field-for-field (camelCase keys)."""
    url = f"{_JAVA_BASE_URL}/internal/practice-sessions/{practice_session_id}/turns"
    async with httpx.AsyncClient(timeout=_SUBMIT_TURN_TIMEOUT_SECONDS) as client:
        response = await client.post(url, headers=_headers(), json=turn)
        response.raise_for_status()
        return response.json()["data"]


async def get_turn_upload_url(practice_session_id: str, turn_order: int) -> dict:
    """GET /internal/practice-sessions/{id}/turns/{turnOrder}/upload-url -- mirrors
    GetTurnUploadUrlUseCase (exam's WPF pattern): Java only mints the presigned S3 PUT URL,
    it never sees the audio bytes -- Python uploads directly (upload_turn_wav below), same as
    WPF does today. Returns {"uploadUrl": ..., "audioRef": ...} (audioRef is the resulting
    public/playable URL, to use as SubmitPracticeTurn.audioUrl once the PUT succeeds)."""
    url = f"{_JAVA_BASE_URL}/internal/practice-sessions/{practice_session_id}/turns/{turn_order}/upload-url"
    async with httpx.AsyncClient(timeout=_UPLOAD_URL_TIMEOUT_SECONDS) as client:
        response = await client.get(url, headers=_headers())
        response.raise_for_status()
        return response.json()["data"]


async def upload_turn_wav(upload_url: str, wav_bytes: bytes) -> None:
    """PUT the WAV directly to S3 via the presigned URL from get_turn_upload_url. Content-Type
    must be exactly "audio/wav" -- it's baked into the presigned signature
    (AwsS3StorageService.presignUpload), a mismatch here fails with SignatureDoesNotMatch."""
    async with httpx.AsyncClient(timeout=_S3_PUT_TIMEOUT_SECONDS) as client:
        response = await client.put(
            upload_url, content=wav_bytes, headers={"Content-Type": "audio/wav"}
        )
        response.raise_for_status()
