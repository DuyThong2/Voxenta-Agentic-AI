"""RabbitMQ configuration using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class RabbitMQSettings(BaseSettings):
    """RabbitMQ settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # RabbitMQ Connection
    RABBITMQ_URI: str = "amqp://guest:guest@localhost:5672/"
    RABBITMQ_HEARTBEAT: int = 1200

    # RabbitMQ Exchange Names
    RABBITMQ_COMPLETED_EXCHANGE: str = "EventSourcing.Events.Lab:PaperIngestionCompletedEvent"
    RABBITMQ_INGEST_EXCHANGE: str = "EventSourcing.Events.Lab:PaperIngestionEvent"

    # RabbitMQ Queue Names
    RABBITMQ_INGEST_QUEUE: str = "paper-ingestion"
    VECTOR_INDEX_QUEUE: str = "vector-indexing"


@lru_cache
def get_rabbitmq_settings() -> RabbitMQSettings:
    """Get cached RabbitMQ settings instance."""
    return RabbitMQSettings()


settings = get_rabbitmq_settings()
