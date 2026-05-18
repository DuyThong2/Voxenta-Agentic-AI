"""
JWT authentication utilities for extracting user information from tokens.

Handles JWT tokens encoded with HS256 (HMAC SHA256) algorithm.
Extracts user ID from ClaimTypes.NameIdentifier claim.
"""

import os
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import logging

from config import JWT_SECRET
from conversation_db import get_conversation

logger = logging.getLogger(__name__)

security = HTTPBearer()

JWT_ISSUER = os.getenv("JWT_ISSUER", "https://localhost:7048")

# allow multiple audiences: "http://localhost:5173,http://127.0.0.1:5173"
aud_raw = os.getenv("JWT_AUDIENCE", "http://localhost:5173")
JWT_AUDIENCES = [x.strip() for x in aud_raw.split(",") if x.strip()]


def decode_jwt_token(token: str) -> dict:
    """
    Decode + verify JWT token.
    - Verify signature (HS256)
    - Verify exp
    - Verify issuer + audience (because your .NET token includes iss/aud)
    """
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=["HS256"],
            audience=JWT_AUDIENCES,  
            issuer=JWT_ISSUER,
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_aud": True,
                "verify_iss": True,
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

    except jwt.InvalidAudienceError:
        logger.warning("Invalid audience. expected=%s", JWT_AUDIENCES)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid audience",
            headers={"WWW-Authenticate": "Bearer"},
        )

    except jwt.InvalidIssuerError:
        logger.warning("Invalid issuer. expected=%s", JWT_ISSUER)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid issuer",
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
    # Try different claim names (in order of likelihood)
    claim_names = [
        "nameid",  # Standard JWT mapping for ClaimTypes.NameIdentifier (sometimes)
        "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/nameidentifier",  # Full .NET claim type ✅
        "NameIdentifier",
        "userId",
        "sub",
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


def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    token = credentials.credentials

    payload = decode_jwt_token(token)
    user_id = extract_user_id_from_payload(payload)

    return user_id


def validate_conversation_id(
    conversation_id: str,
    user_id: str,
    request: Request
) -> str:
    """
    Validate that conversation_id belongs to the authenticated user.
    """
    pool = getattr(request.app.state, "pg_pool", None)
    if pool is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database pool not initialized"
        )

    conversation = get_conversation(pool, conversation_id, user_id)

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found or access denied"
        )

    return conversation_id
