"""Схемы входных и выходных данных серверного API."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .constants import EMOTIONS


def validate_timezone_name(value: str) -> str:
    try:
        ZoneInfo(value)
    except ZoneInfoNotFoundError as exc:
        raise ValueError("Неизвестный часовой пояс") from exc
    return value


class RegistrationCodeCreate(BaseModel):
    expires_minutes: int = Field(default=30, ge=5, le=1440)


class RegistrationCodeResponse(BaseModel):
    code: str
    expires_at: datetime


class ClientRegisterRequest(BaseModel):
    code: str = Field(min_length=8, max_length=256)
    display_name: str = Field(min_length=1, max_length=120)
    device_name: str = Field(min_length=1, max_length=160)
    timezone_name: str = Field(default="UTC", max_length=100)
    client_version: str = Field(min_length=1, max_length=64)

    _validate_timezone = field_validator("timezone_name")(validate_timezone_name)


class ClientRegisterResponse(BaseModel):
    user_id: uuid.UUID
    device_id: uuid.UUID
    token: str
    display_name: str


class UserSwitchRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=120)
    timezone_name: str = Field(default="UTC", max_length=100)

    _validate_timezone = field_validator("timezone_name")(validate_timezone_name)


class HeartbeatRequest(BaseModel):
    client_version: str = Field(min_length=1, max_length=64)
    model_version: str | None = Field(default=None, max_length=128)
    model_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    monitoring_active: bool


class EmotionEventInput(BaseModel):
    message_id: uuid.UUID
    occurred_at: datetime
    timezone_name: str = Field(max_length=100)
    segment_duration_sec: float = Field(gt=0.0, le=60.0)
    speech_duration_sec: float = Field(gt=0.0, le=60.0)
    predicted_emotion: Literal["joy", "sadness", "anger", "surprise", "calm", "disgust", "fear"]
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    quality: dict[str, Any] = Field(default_factory=dict)
    playback_active: bool = False
    output_is_headphones: bool = False
    filtered_before_send_count: int = Field(default=0, ge=0)
    model_version: str = Field(min_length=1, max_length=128)
    model_sha256: str = Field(min_length=64, max_length=64)
    client_version: str = Field(min_length=1, max_length=64)
    processing_time_ms: float = Field(ge=0.0, le=60000.0)

    @field_validator("occurred_at")
    @classmethod
    def validate_aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at должен содержать часовой пояс")
        return value

    _validate_timezone = field_validator("timezone_name")(validate_timezone_name)

    @model_validator(mode="after")
    def validate_probabilities(self) -> "EmotionEventInput":
        if set(self.probabilities) != set(EMOTIONS):
            raise ValueError("Вероятности должны содержать все семь эмоций")
        values = list(self.probabilities.values())
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("Вероятности должны находиться в диапазоне от 0 до 1")
        total = sum(values)
        if not 0.98 <= total <= 1.02:
            raise ValueError("Сумма вероятностей должна быть близка к 1")
        if self.predicted_emotion != max(self.probabilities, key=self.probabilities.get):
            raise ValueError("predicted_emotion не соответствует максимальной вероятности")
        if self.speech_duration_sec > self.segment_duration_sec:
            raise ValueError("Длительность речи не может превышать длительность фрагмента")
        return self


class EmotionEventBatchRequest(BaseModel):
    events: list[EmotionEventInput] = Field(min_length=1, max_length=500)


class EmotionEventBatchResponse(BaseModel):
    accepted: int
    duplicates: int
    rejected: int
    accepted_message_ids: list[uuid.UUID]
    duplicate_message_ids: list[uuid.UUID]
    rejected_message_ids: list[uuid.UUID]


class ModelMetadataResponse(BaseModel):
    version: str
    display_name: str
    model_type: str
    sha256: str
    signature: str
    config: dict[str, Any]
    thresholds: dict[str, Any]
    download_url: str


class ModelAckRequest(BaseModel):
    sha256: str = Field(min_length=64, max_length=64)
    success: bool
    error_text: str | None = Field(default=None, max_length=2000)


class AggregateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user_id: uuid.UUID
    period_type: str
    period_start_utc: datetime
    period_end_utc: datetime
    timezone_name: str
    dominant_emotion: str | None
    probabilities: dict[str, float]
    emotion_shares: dict[str, float]
    event_count: int
    speech_duration_sec: float
    average_confidence: float
    filtered_count: int
    volatility: float
    has_data: bool


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    display_name: str
    timezone_name: str
    is_active: bool
    created_at: datetime
    deleted_at: datetime | None


class DeviceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    device_name: str
    client_version: str
    model_version: str | None
    model_sha256: str | None
    last_seen_at: datetime | None
    monitoring_active: bool
    is_blocked: bool
    is_online: bool
    created_at: datetime


class EventAdminResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    message_id: uuid.UUID
    user_id: uuid.UUID
    device_id: uuid.UUID
    occurred_at: datetime
    timezone_name: str
    speech_duration_sec: float
    predicted_emotion: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str
    model_is_current: bool
    deleted_at: datetime | None
    deletion_reason: str | None


class ModelAdminResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    version: str
    display_name: str
    model_type: str
    sha256: str
    is_published: bool
    created_at: datetime
    published_at: datetime | None


class EventDeleteRequest(BaseModel):
    reason: str = Field(min_length=3, max_length=1000)
