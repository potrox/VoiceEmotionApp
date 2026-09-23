"""Проверки расчёта временных агрегатов."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest

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


class AggregationTests(unittest.TestCase):
    def test_hour_requires_thirty_seconds(self):
        events = [make_event("joy", 0.9, index) for index in range(5)]
        result = calculate_aggregate(events, "hour", 30.0)
        self.assertFalse(result["has_data"])
        events.append(make_event("joy", 0.9, 5))
        result = calculate_aggregate(events, "hour", 30.0)
        self.assertTrue(result["has_data"])
        self.assertEqual(result["dominant_emotion"], "joy")

    def test_period_bounds_are_calendar_based(self):
        occurred = datetime(2026, 1, 10, 10, 45, tzinfo=timezone.utc)
        start, end = period_bounds(occurred, "UTC", "hour")
        self.assertEqual(start.minute, 0)
        self.assertEqual(end - start, timedelta(hours=1))

    def test_four_class_events_aggregate_without_key_errors(self):
        event = make_event("joy", 0.9, 0)
        event.probabilities = {key: event.probabilities[key] for key in ("joy", "sadness", "anger", "calm")}
        result = calculate_aggregate([event], "hour", 0.0)
        self.assertIn("joy", result["probabilities"])
        self.assertEqual(result["probabilities"]["fear"], 0.0)


if __name__ == "__main__":
    unittest.main()
