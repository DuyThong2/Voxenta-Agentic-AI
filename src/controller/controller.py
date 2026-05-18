"""
API controller endpoints for the chat application.

The graph + Postgres checkpointer handle conversation history.
"""

from typing import List
from fastapi import APIRouter, HTTPException, Request, Depends
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional

from config import JWT_SECRET
from utils import new_id, now_iso
from auth import get_current_user_id, validate_conversation_id


from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
import os


# Create router
router = APIRouter()
logger = logging.getLogger(__name__)

security = HTTPBearer()

@router.get("/health")
def health():
    """
    Liveness probe.
    Used by Docker / Kubernetes / load balancer.
    """
    return {
        "status": "ok",
        "service": "viren-api",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/_debug/jwt-verify")
def debug_jwt_verify(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials

    # đọc expected từ env (giống auth.py)
    issuer = os.getenv("JWT_ISSUER", "https://localhost:7048")
    aud_raw = os.getenv("JWT_AUDIENCE", "http://localhost:5173")
    audiences = [x.strip() for x in aud_raw.split(",") if x.strip()]

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=["HS256"],
            audience=audiences,  
            issuer=issuer,       
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_aud": True,
                "verify_iss": True,
            }
        )
        return {
            "ok": True,
            "payload": payload,
            "expected_aud": audiences,
            "expected_iss": issuer,
            "secret_len": len(JWT_SECRET or "")
        }
    except Exception as e:
        raw = jwt.decode(token, options={"verify_signature": False})
        return {
            "ok": False,
            "error": str(e),
            "expected_aud": audiences,
            "expected_iss": issuer,
            "token_aud": raw.get("aud"),
            "token_iss": raw.get("iss"),
            "secret_len": len(JWT_SECRET or ""),
            "secret_preview": (JWT_SECRET[:4] + "..." + JWT_SECRET[-4:]) if JWT_SECRET else None
        }


@router.get("/_debug/jwt")
def debug_jwt(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    DEV ONLY:
    Decode JWT without verifying signature to inspect payload.
    """
    token = credentials.credentials

    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        return {
            "ok": True,
            "payload": payload
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }



