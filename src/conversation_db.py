"""Conversation database utilities."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_conversation(pool, conversation_id: str, user_id: str) -> Optional[dict]:
    """Get a conversation by ID and user ID."""
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM conversations WHERE id = %s AND user_id = %s",
                    (conversation_id, user_id),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                cols = [desc[0] for desc in cur.description]
                return dict(zip(cols, row))
    except Exception:
        logger.exception("[conversation_db] get_conversation failed")
        return None
