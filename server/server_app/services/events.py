"""Проверка и сохранение пакетов событий распознавания."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..models import AggregationJob, Device, EmotionEvent, ModelVersion
from ..schemas import EmotionEventInput


def minimum_confidence(model: ModelVersion | None, item: EmotionEventInput) -> float:
    if model is None:
        return 0.0
    thresholds = model.thresholds_json or {}
    threshold = float(thresholds.get("default", 0.0))
    per_emotion = thresholds.get("per_emotion") or {}
    threshold = max(threshold, float(per_emotion.get(item.predicted_emotion, 0.0)))
    if item.segment_duration_sec < 3.0:
        threshold += float(thresholds.get("short_segment_bonus", 0.0))
    if item.playback_active and not item.output_is_headphones:
        threshold += float(thresholds.get("speaker_playback_bonus", 0.0))
    return min(max(threshold, 0.0), 1.0)


def insert_event(
    db: Session,
    device: Device,
    item: EmotionEventInput,
    active_model: ModelVersion | None,
    event_model: ModelVersion | None,
) -> tuple[EmotionEvent | None, bool, bool]:
    if item.confidence < minimum_confidence(event_model, item):
        return None, False, True
    model_is_current = bool(
        active_model is not None
        and item.model_version == active_model.version
        and item.model_sha256 == active_model.sha256
    )
    values = {
        "message_id": item.message_id,
        "user_id": device.user_id,
        "device_id": device.id,
        "occurred_at": item.occurred_at.astimezone(timezone.utc),
        "timezone_name": item.timezone_name,
        "segment_duration_sec": item.segment_duration_sec,
        "speech_duration_sec": item.speech_duration_sec,
        "predicted_emotion": item.predicted_emotion,
        "confidence": item.confidence,
        "probabilities": item.probabilities,
        "quality": item.quality,
        "playback_active": item.playback_active,
        "output_is_headphones": item.output_is_headphones,
        "filtered_before_send_count": item.filtered_before_send_count,
        "model_version": item.model_version,
        "model_sha256": item.model_sha256,
        "model_is_current": model_is_current,
        "client_version": item.client_version,
        "processing_time_ms": item.processing_time_ms,
    }
    statement = insert(EmotionEvent).values(**values).on_conflict_do_nothing(index_elements=[EmotionEvent.message_id]).returning(EmotionEvent.id)
    event_id = db.execute(statement).scalar_one_or_none()
    if event_id is None:
        return None, True, False
    event = db.get(EmotionEvent, event_id)
    if event is None:
        return None, False, True
    db.add(AggregationJob(user_id=device.user_id, event_id=event.id))
    return event, False, False


def event_exists(db: Session, message_id) -> bool:
    return db.scalar(select(EmotionEvent.id).where(EmotionEvent.message_id == message_id)) is not None


def touch_device(db: Session, device: Device, client_version: str, model_version: str, model_sha256: str) -> None:
    device.last_seen_at = datetime.now(timezone.utc)
    device.client_version = client_version
    device.model_version = model_version
    device.model_sha256 = model_sha256
