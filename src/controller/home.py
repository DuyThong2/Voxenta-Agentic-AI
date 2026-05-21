from datetime import datetime
from fastapi import APIRouter, Request

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
