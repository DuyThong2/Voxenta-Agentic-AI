"""Environment and path utilities for loading root .env configuration."""

from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv


def get_project_root() -> Path:
    """Return the repository root directory from inside the src package."""
    return Path(__file__).resolve().parent.parent.parent


def load_root_dotenv(
    env_file: Optional[Union[str, Path]] = None,
    override: bool = False,
) -> bool:
    """Load the root .env file from the repository root.

    This helper is reusable across modules and avoids duplicating the root
    path calculation in multiple files.
    """
    path = Path(env_file) if env_file else get_project_root() / ".env"
    return load_dotenv(dotenv_path=path, override=override)
