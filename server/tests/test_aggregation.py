"""Проверки расчёта временных агрегатов."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from server_app.services.aggregation_core import calculate_aggregate, period_bounds


PROBABILITIES = {
    "joy": 0.7,
    "sadness": 0.05,
    "anger": 0.05,
    "surprise": 0.05,
    "calm": 0.1,
    "disgust": 0.025,
    "fear": 0.025,
}


def make_event(emotion: str, confidence: float, second: int):
    values = dict(PROBABILITIES)
    values["joy"] = 0.05
    values[emotion] = 0.7
    return SimpleNamespace(
        occurred_at=datetime(2026, 1, 1, 12, 0, second, tzinfo=timezone.utc),
        speech_duration_sec=5.0,
        confidence=confidence,
        predicted_emotion=emotion,
        probabilities=values,
        filtered_before_send_count=0,
    )


def test_hour_requires_thirty_seconds():
    events = [make_event("joy", 0.9, index) for index in range(5)]
    result = calculate_aggregate(events, "hour", 30.0)
    assert result["has_data"] is False
    events.append(make_event("joy", 0.9, 5))
    result = calculate_aggregate(events, "hour", 30.0)
    assert result["has_data"] is True
    assert result["dominant_emotion"] == "joy"


def test_period_bounds_are_calendar_based():
    occurred = datetime(2026, 1, 10, 10, 45, tzinfo=timezone.utc)
    start, end = period_bounds(occurred, "UTC", "hour")
    assert start.minute == 0
    assert end - start == timedelta(hours=1)
