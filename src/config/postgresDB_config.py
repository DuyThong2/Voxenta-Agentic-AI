"""PostgreSQL configuration using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class PostgresSettings(BaseSettings):
    """PostgreSQL settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # PostgreSQL Connection
    PG_HOST: str = "localhost"
    PG_PORT: int = 5433
    PG_USER: str = "vox"
    PG_PASSWORD: str = "Vox_password1!"
    PG_DATABASE: str = "vox_langgraph"
    PG_SSLMODE: str = "disable"

    @property
    def PG_URI(self) -> str:
        return (
            f"postgresql://{self.PG_USER}:{self.PG_PASSWORD}"
            f"@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DATABASE}"
            f"?sslmode={self.PG_SSLMODE}"
        )


@lru_cache
def get_postgres_settings() -> PostgresSettings:
    """Get cached PostgreSQL settings instance."""
    return PostgresSettings()


settings = get_postgres_settings()
