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
    KAFKA_AUTO_OFFSET_RESET: str = "earliest"

    # 2 topic cho cả luồng exam-evaluation — Failed dùng chung topic completed,
    # phân biệt bằng EventEnvelope.event_type (xem src/dtos/), không cần thêm topic DLQ riêng.
    KAFKA_EXAM_REQUEST_TOPIC: str = "exam-attempt-evaluation-requested"
    KAFKA_EXAM_COMPLETED_TOPIC: str = "exam-attempt-evaluation-completed"
    KAFKA_PRACTICE_REQUEST_TOPIC: str = "practice-attempt-evaluation-requested"
    KAFKA_PRACTICE_COMPLETED_TOPIC: str = "practice-attempt-evaluation-completed"
    KAFKA_ANSWER_TURN_TOPIC: str = "answer-turns-recorded"
    KAFKA_QUESTION_ASSET_ANALYSIS_REQUEST_TOPIC: str = "question-asset-analysis-requested"
    KAFKA_QUESTION_ASSET_ANALYSIS_COMPLETED_TOPIC: str = "question-asset-analysis-completed"
    KAFKA_EXAM_FORCE_END_TOPIC: str = "exam-attempt-force-end-requested"
    KAFKA_EXAM_CONSUMER_GROUP: str = "exam-attempt-evaluation"
    KAFKA_PRACTICE_CONSUMER_GROUP: str = "practice-attempt-evaluation"
    KAFKA_EXAM_CONSUMER_CONCURRENCY: int = 4
    KAFKA_EXAM_FORCE_END_CONSUMER_GROUP: str = "exam-attempt-force-end"
    KAFKA_QUESTION_ASSET_ANALYSIS_CONSUMER_GROUP: str = "question-asset-analysis"
    KAFKA_MAX_RETRY: int = 3


@lru_cache
def get_kafka_settings() -> KafkaSettings:
    """Get cached Kafka settings instance."""
    return KafkaSettings()


settings = get_kafka_settings()
