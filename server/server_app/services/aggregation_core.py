"""Чистые расчёты агрегатов, не зависящие от базы данных и SQLAlchemy."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Mapping, Protocol, Sequence
from zoneinfo import ZoneInfo

from ..constants import EMOTIONS

PERIOD_TYPES = ("hour", "day", "week", "month")


class EmotionEventData(Protocol):

    occurred_at: datetime
    speech_duration_sec: float
    confidence: float
    predicted_emotion: str
    probabilities: Mapping[str, float]
    filtered_before_send_count: int


def period_bounds(
    occurred_at: datetime, timezone_name: str, period_type: str
) -> tuple[datetime, datetime]:
    if period_type not in PERIOD_TYPES:
        raise ValueError("Неизвестный тип периода")
    local_time = occurred_at.astimezone(ZoneInfo(timezone_name))
    if period_type == "hour":
        period_start = local_time.replace(minute=0, second=0, microsecond=0)
        period_end = period_start + timedelta(hours=1)
    elif period_type == "day":
        period_start = local_time.replace(hour=0, minute=0, second=0, microsecond=0)
        period_end = period_start + timedelta(days=1)
    elif period_type == "week":
        day_start = local_time.replace(hour=0, minute=0, second=0, microsecond=0)
        period_start = day_start - timedelta(days=day_start.weekday())
        period_end = period_start + timedelta(days=7)
    else:
        period_start = local_time.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        period_end = (
            period_start.replace(year=period_start.year + 1, month=1)
            if period_start.month == 12
            else period_start.replace(month=period_start.month + 1)
        )
    return period_start.astimezone(timezone.utc), period_end.astimezone(timezone.utc)


def calculate_aggregate(
    events: Sequence[EmotionEventData],
    period_type: str,
    minimum_hour_speech_seconds: float,
) -> dict:
    total_speech_seconds = sum(
        max(event.speech_duration_sec, 0.0) for event in events
    )
    required_speech_seconds = (
        minimum_hour_speech_seconds if period_type == "hour" else 0.0
    )
    has_data = bool(events) and total_speech_seconds >= required_speech_seconds
    weighted_probabilities = defaultdict(float)
    predicted_weights = defaultdict(float)
    confidence_duration_sum = 0.0
    total_weight = 0.0
    filtered_count = 0
    ordered_emotions: list[str] = []

    for event in sorted(events, key=lambda item: item.occurred_at):
        duration_seconds = max(event.speech_duration_sec, 0.001)
        event_weight = duration_seconds * max(event.confidence, 0.001)
        total_weight += event_weight
        confidence_duration_sum += event.confidence * duration_seconds
        filtered_count += event.filtered_before_send_count
        ordered_emotions.append(event.predicted_emotion)
        predicted_weights[event.predicted_emotion] += event_weight
        for emotion in EMOTIONS:
            weighted_probabilities[emotion] += (
                float(event.probabilities.get(emotion, 0.0)) * event_weight
            )

    if total_weight > 0.0:
        probabilities = {
            emotion: weighted_probabilities[emotion] / total_weight
            for emotion in EMOTIONS
        }
        predicted_total = sum(predicted_weights.values())
        emotion_shares = {
            emotion: predicted_weights[emotion] / predicted_total
            if predicted_total
            else 0.0
            for emotion in EMOTIONS
        }
        dominant_emotion = max(probabilities, key=probabilities.get) if has_data else None
    else:
        probabilities = {emotion: 0.0 for emotion in EMOTIONS}
        emotion_shares = {emotion: 0.0 for emotion in EMOTIONS}
        dominant_emotion = None

    emotion_changes = sum(
        left_emotion != right_emotion
        for left_emotion, right_emotion in zip(
            ordered_emotions, ordered_emotions[1:]
        )
    )
    volatility = (
        emotion_changes / (len(ordered_emotions) - 1)
        if len(ordered_emotions) > 1
        else 0.0
    )
    confidence_duration = max(total_speech_seconds, 0.001)
    return {
        "dominant_emotion": dominant_emotion,
        "probabilities": {
            emotion: round(value, 8) for emotion, value in probabilities.items()
        },
        "emotion_shares": {
            emotion: round(value, 8) for emotion, value in emotion_shares.items()
        },
        "event_count": len(events),
        "speech_duration_sec": round(total_speech_seconds, 6),
        "average_confidence": (
            round(confidence_duration_sum / confidence_duration, 8) if events else 0.0
        ),
        "filtered_count": filtered_count,
        "volatility": round(volatility, 8),
        "has_data": has_data,
    }
