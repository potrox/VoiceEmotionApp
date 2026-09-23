"""Загрузка и проверка переменных окружения сервера."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "VoiceEmotionServer"
    app_version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    database_url: str = Field(alias="DATABASE_URL")
    admin_api_key: str = Field(alias="ADMIN_API_KEY")
    token_secret: str = Field(alias="TOKEN_SECRET")
    model_signing_key: str = Field(alias="MODEL_SIGNING_KEY")
    data_encryption_key: str = Field(alias="DATA_ENCRYPTION_KEY")
    models_dir: str = Field(default="/data/models", alias="SERVER_MODELS_DIR")
    event_retention_days: int = Field(default=14, alias="EVENT_RETENTION_DAYS")
    aggregate_retention_days: int = Field(default=365, alias="AGGREGATE_RETENTION_DAYS")
    audit_retention_days: int = Field(default=365, alias="AUDIT_RETENTION_DAYS")
    device_online_seconds: int = Field(default=120, alias="DEVICE_ONLINE_SECONDS")
    minimum_hour_speech_seconds: float = Field(default=30.0, alias="MINIMUM_HOUR_SPEECH_SECONDS")
    worker_poll_seconds: float = Field(default=2.0, alias="WORKER_POLL_SECONDS")
    scheduler_poll_seconds: float = Field(default=60.0, alias="SCHEDULER_POLL_SECONDS")
    registration_code_minutes: int = Field(default=30, alias="REGISTRATION_CODE_MINUTES")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator("admin_api_key", "token_secret", "model_signing_key", "data_encryption_key")
    @classmethod
    def validate_secret(cls, value: str) -> str:
        if len(value.strip()) < 32:
            raise ValueError("Секрет должен содержать не менее 32 символов")
        return value.strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
