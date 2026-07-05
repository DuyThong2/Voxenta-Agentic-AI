"""S3 configuration using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class S3Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    STORAGE_PROVIDER: str = "s3"
    S3_BUCKET_NAME: str = Field(
        default="",
        validation_alias=AliasChoices("S3_BUCKET_NAME", "AWS_S3_BUCKET"),
    )
    S3_ENDPOINT_URL: Optional[str] = None
    AWS_REGION: str = "ap-southeast-1"
    AWS_S3_KEY_PREFIX: Optional[str] = None
    AWS_S3_PUBLIC_BASE_URL: Optional[str] = None
    AWS_S3_PATH_STYLE: bool = False


@lru_cache
def get_s3_settings() -> S3Settings:
    return S3Settings()


settings = get_s3_settings()
