"""ChromaDB configuration using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class ChromaSettings(BaseSettings):
    """ChromaDB settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ChromaDB Connection
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001
    CHROMA_COLLECTION: str = "product_details_openai"

    # OpenAI Embedding
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"


@lru_cache
def get_chroma_settings() -> ChromaSettings:
    """Get cached ChromaDB settings instance."""
    return ChromaSettings()


settings = get_chroma_settings()
