"""Kafka configuration using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class KafkaSettings(BaseSettings):
    """Kafka settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_CLIENT_ID: str = "vox-agents"
    KAFKA_CONSUMER_GROUP: str = "vector-indexing"
    KAFKA_AUTO_OFFSET_RESET: str = "earliest"

    KAFKA_COMPLETED_TOPIC: str = "paper-ingestion-completed"
    KAFKA_INGEST_TOPIC: str = "paper-ingestion"
    KAFKA_VECTOR_INDEX_TOPIC: str = "vector-indexing"
    # 2 topic cho cả luồng exam-evaluation — Failed dùng chung topic completed,
    # phân biệt bằng EventEnvelope.event_type (xem src/dtos/), không cần thêm topic DLQ riêng.
    KAFKA_EXAM_REQUEST_TOPIC: str = "exam-attempt-evaluation-requested"
    KAFKA_EXAM_COMPLETED_TOPIC: str = "exam-attempt-evaluation-completed"
    KAFKA_EXAM_CONSUMER_GROUP: str = "exam-attempt-evaluation"
    KAFKA_MAX_RETRY: int = 3


@lru_cache
def get_kafka_settings() -> KafkaSettings:
    """Get cached Kafka settings instance."""
    return KafkaSettings()


settings = get_kafka_settings()
