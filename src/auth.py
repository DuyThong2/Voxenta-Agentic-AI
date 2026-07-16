"""
JWT authentication utilities for extracting user information from tokens.

Expected payload shape:
- sub: UUID of the authenticated user
- userId: UUID of the authenticated user
- email: user email
- roles: list of role codes
"""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional

import jwt
import logging
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import JWT_SECRET
from conversation_db import get_conversation

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


@dataclass(slots=True)
class CurrentUser:
    user_id: str
    subject: Optional[str]
    email: Optional[str]
    roles: list[str]
    claims: dict
    token: str


_current_user_ctx: ContextVar[Optional[CurrentUser]] = ContextVar("current_user", default=None)


def decode_jwt_token(token: str) -> dict:
    """
    Decode and verify JWT token.
    - Verify signature (HS256)
    - Verify exp
    - Do not verify issuer/audience because current tokens do not include iss/aud
    """
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
        return payload

    except jwt.ExpiredSignatureError:
        logger.warning("JWT token has expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )

    except jwt.InvalidTokenError as e:
        logger.warning("Invalid JWT token: %s", repr(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    except Exception as e:
        logger.error("Error decoding JWT token: %s", repr(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to decode token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def extract_user_id_from_payload(payload: dict) -> str:
    # Prioritize claims from the current auth service, then keep legacy fallbacks.
    claim_names = [
        "userId",
        "sub",
        "nameid",
        "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/nameidentifier",
        "NameIdentifier",
    ]

    for claim_name in claim_names:
        user_id = payload.get(claim_name)
        if user_id:
            return str(user_id)

    logger.error("User ID not found in JWT payload. Available claims: %s", list(payload.keys()))
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="User ID not found in token",
        headers={"WWW-Authenticate": "Bearer"},
    )


def build_current_user(token: str) -> CurrentUser:
    payload = decode_jwt_token(token)
    user_id = extract_user_id_from_payload(payload)

    roles = payload.get("roles") or []
    if not isinstance(roles, list):
        roles = [str(roles)]
    else:
        roles = [str(role) for role in roles]

    return CurrentUser(
        user_id=user_id,
        subject=str(payload["sub"]) if payload.get("sub") is not None else None,
        email=str(payload["email"]) if payload.get("email") is not None else None,
        roles=roles,
        claims=payload,
        token=token,
    )


def set_current_user_context(user: Optional[CurrentUser]) -> None:
    _current_user_ctx.set(user)


def get_current_user_context() -> Optional[CurrentUser]:
    return _current_user_ctx.get()


def _extract_bearer_token_from_request(request: Request) -> Optional[str]:
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return None

    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token.strip()


def get_current_user_from_request(request: Request, *, required: bool = True) -> Optional[CurrentUser]:
    user = getattr(request.state, "current_user", None)
    if user is not None:
        return user

    token = _extract_bearer_token_from_request(request)
    if token:
        try:
            user = build_current_user(token)
        except HTTPException:
            if required:
                raise
            return None

        request.state.current_user = user
        request.state.current_user_id = user.user_id
        set_current_user_context(user)
        return user

    if required:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return None


def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = build_current_user(credentials.credentials)
    set_current_user_context(user)
    return user.user_id


def validate_conversation_id(
    conversation_id: str,
    user_id: str,
    request: Request,
) -> str:
    """
    Validate that conversation_id belongs to the authenticated user.
    """
    pool = getattr(request.app.state, "pg_pool", None)
    if pool is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database pool not initialized",
        )

    conversation = get_conversation(pool, conversation_id, user_id)

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found or access denied",
        )

    return conversation_id
