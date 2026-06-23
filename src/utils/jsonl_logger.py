"""Local-only JSONL request/response logging for debugging while running on
a dev machine. Best-effort: a write failure here must never break the
request/publish it's attached to.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from utils.env import get_project_root

logger = logging.getLogger(__name__)

_LOG_DIR = get_project_root() / "logs"


def append_jsonl(filename: str, record: Dict[str, Any]) -> None:
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(_LOG_DIR / filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:
        logger.warning("[jsonl_logger] failed to write %s", filename, exc_info=True)
