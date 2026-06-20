import logging

import jwt
from fastapi import APIRouter, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from auth import get_current_user_from_request
from config import JWT_SECRET


router = APIRouter(prefix="/auth", tags=["Auth"])
logger = logging.getLogger(__name__)
security = HTTPBearer()


@router.get("/_debug/jwt-verify")
def debug_jwt_verify(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=["HS256"],
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_aud": False,
                "verify_iss": False,
                "require": ["exp"],
            },
        )
        return {
            "ok": True,
            "payload": payload,
            "secret_len": len(JWT_SECRET or ""),
        }
    except Exception as e:
        raw = jwt.decode(token, options={"verify_signature": False})
        return {
            "ok": False,
            "error": str(e),
            "token_sub": raw.get("sub"),
            "token_user_id": raw.get("userId"),
            "token_email": raw.get("email"),
            "token_roles": raw.get("roles"),
            "token_aud": raw.get("aud"),
            "token_iss": raw.get("iss"),
            "secret_len": len(JWT_SECRET or ""),
            "secret_preview": (JWT_SECRET[:4] + "..." + JWT_SECRET[-4:]) if JWT_SECRET else None,
        }


@router.get("/_debug/jwt")
def debug_jwt(
    credentials: HTTPAuthorizationCredentials = Depends(security),
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
            "payload": payload,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
        }


@router.get("/test")
async def test():
    return {"message": "Auth controller loaded successfully"}


@router.get("/me")
def get_me(request: Request):
    user = get_current_user_from_request(request)
    return {
        "user_id": user.user_id,
        "subject": user.subject,
        "email": user.email,
        "roles": user.roles,
        "claims": user.claims,
    }
