"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings sourced from .env / process env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "PIEmaker Backend"
    app_env: str = Field(default="development")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    database_url: str = Field(
        default="postgresql+psycopg2://piemaker:piemaker@localhost:5432/piemaker"
    )
    redis_url: str = Field(default="redis://localhost:6379/0")

    s3_endpoint_url: str | None = None
    s3_bucket: str = Field(default="piemaker-uploads")
    s3_access_key: str | None = None
    s3_secret_key: str | None = None

    mlflow_tracking_uri: str = Field(default="http://localhost:5000")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
