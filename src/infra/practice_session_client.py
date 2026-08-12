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
# Must comfortably exceed Java's worst-case latency for this call, not just the common case:
# ResolveNextPracticeQuestionUseCase can fall through to bậc 4 (online LLM generation, up to
# PracticeGenerationProperties.onlineBudget = 20s) AND holds a FOR UPDATE row lock on the
# session for the whole transaction. A retry that fires while attempt 1 is still mid-generation
# just blocks on that same lock for the remainder of attempt 1's run -- an 8s budget left almost
# no room for that, causing spurious next_question_call_failed / session-ends on the completely
# normal "brand new topic, question bank still empty" cold-start path.
_NEXT_QUESTION_TIMEOUT_SECONDS = 30.0
_SUBMIT_TURN_TIMEOUT_SECONDS = 8.0
_UPLOAD_URL_TIMEOUT_SECONDS = 5.0
_S3_PUT_TIMEOUT_SECONDS = 15.0
# Nhip tim phai NGAN: no chay dinh ky, treo lau chi lam nghen vong lap.
_HEARTBEAT_TIMEOUT_SECONDS = 5.0


def _headers() -> dict:
    return {"X-Internal-Secret": _INTERNAL_SECRET, "Content-Type": "application/json"}


async def request_next_question(practice_session_id: str) -> dict:
    """POST /internal/practice-sessions/{id}/next-question -- returns Java's
    {"message":..., "data": {"status": "ok"|"no_more_questions", "reason": ..., "question": {...}}}.
    Raises httpx.HTTPStatusError/TimeoutException on failure -- caller decides how to treat that
    (see PracticeQuestionSessionCoordinator.resolve_and_push_next_question).

    Retries once on a timeout specifically: Java's ResolveNextPracticeQuestionUseCase is
    @Transactional and may have already committed a new paper item by the time the client-side
    timeout fires (slow network/DB, not a Java-side failure). A retry is safe because that use
    case is now idempotent -- if the previous call's item was never delivered to the student, a
    retry returns that SAME item instead of picking a new one (see the "existsResponse" check in
    ResolveNextPracticeQuestionUseCase.execute), so this can't create an orphaned duplicate."""
    url = f"{_JAVA_BASE_URL}/internal/practice-sessions/{practice_session_id}/next-question"
    async with httpx.AsyncClient(timeout=_NEXT_QUESTION_TIMEOUT_SECONDS) as client:
        try:
            response = await client.post(url, headers=_headers())
        except httpx.TimeoutException:
            logger.warning(
                "[practice_session_client] next-question timed out, retrying once "
                "practice_session_id=%s",
                practice_session_id,
            )
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


async def send_heartbeat(practice_session_id: str) -> None:
    """POST /internal/practice-sessions/{id}/heartbeat -- bao Java rang phien VAN CON SONG.

    Vi sao agents gui chu khong phai app: app da giu WebSocket toi day va bom audio lien tuc, nen
    o day biet tung giay phien con song hay khong. Timer trong app thi van chay du mang da rot,
    va Android bop timer khi app xuong nen -- ca hai deu cho tin hieu sai.

    NUOT MOI LOI co chu dich. Day la tin hieu song, khong phai lenh nghiep vu: nem ra se giet
    vong lap nhip (roi phien chet dung vi thu le ra phai giu no song), con phien vua ket thuc
    binh thuong ma nhip cuoi con dang bay thi khong co gi de xu ly ca.
    """
    url = f"{_JAVA_BASE_URL}/internal/practice-sessions/{practice_session_id}/heartbeat"
    try:
        async with httpx.AsyncClient(timeout=_HEARTBEAT_TIMEOUT_SECONDS) as client:
            await client.post(url, headers=_headers())
    except Exception as exc:
        logger.warning(
            "[practice_session_client] heartbeat that bai practice_session_id=%s: %s",
            practice_session_id, exc,
        )


async def upload_turn_wav(upload_url: str, wav_bytes: bytes) -> None:
    """PUT the WAV directly to S3 via the presigned URL from get_turn_upload_url. Content-Type
    must be exactly "audio/wav" -- it's baked into the presigned signature
    (AwsS3StorageService.presignUpload), a mismatch here fails with SignatureDoesNotMatch."""
    async with httpx.AsyncClient(timeout=_S3_PUT_TIMEOUT_SECONDS) as client:
        response = await client.put(
            upload_url, content=wav_bytes, headers={"Content-Type": "audio/wav"}
        )
        response.raise_for_status()
