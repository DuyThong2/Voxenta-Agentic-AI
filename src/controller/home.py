from datetime import datetime
import asyncio

from fastapi import APIRouter

from infra.voice_live_client import VoiceLiveClient

router = APIRouter()

@router.get("/health")
def health():
    """
    Liveness probe.
    Used by Docker / Kubernetes / load balancer.
    """
    return {
        "status": "ok",
        "service": "vox-api",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/internal/voice-live/readiness", status_code=204)
async def voice_live_readiness() -> None:
    """Open and close a real Voice Live session before practice starts."""

    async def ignore_event(_event) -> None:
        return None

    client = VoiceLiveClient(on_event=ignore_event)
    try:
        await asyncio.wait_for(client.start(), timeout=8)
    finally:
        await client.close()
