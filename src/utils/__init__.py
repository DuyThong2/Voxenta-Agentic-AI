"""Utility functions and helpers."""

import uuid
from datetime import datetime, timezone

from .env import get_project_root, load_root_dotenv
from .transcript_selector import select_text_for_language_scoring, build_scoring_metadata

__all__ = [
    "new_id",
    "now_iso",
    "get_project_root",
    "load_root_dotenv",
    "select_text_for_language_scoring",
    "build_scoring_metadata",
]


def new_id() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


def now_iso() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()
