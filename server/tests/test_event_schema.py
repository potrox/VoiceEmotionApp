"""Compatibility checks for current and legacy emotion events."""

from datetime import datetime, timezone
import unittest
import uuid

from fastapi import HTTPException

from server_app.schemas import EmotionEventInput
from server_app.services.models_service import validate_model_version


class EventSchemaTests(unittest.TestCase):
    def make_event(self, probabilities):
        return EmotionEventInput(
            message_id=uuid.uuid4(),
            occurred_at=datetime.now(timezone.utc),
            timezone_name="UTC",
            segment_duration_sec=5.0,
            speech_duration_sec=5.0,
            predicted_emotion="joy",
            confidence=0.7,
            probabilities=probabilities,
            model_version="v11",
            model_sha256="0" * 64,
            client_version="test",
            processing_time_ms=10.0,
        )

    def test_four_class_event(self):
        event = self.make_event({"joy": 0.7, "sadness": 0.1, "anger": 0.1, "calm": 0.1})
        self.assertEqual(event.predicted_emotion, "joy")

    def test_seven_class_legacy_event(self):
        event = self.make_event({
            "joy": 0.7, "sadness": 0.05, "anger": 0.05, "surprise": 0.05,
            "calm": 0.1, "disgust": 0.025, "fear": 0.025,
        })
        self.assertEqual(len(event.probabilities), 7)

    def test_uploaded_model_version_cannot_contain_path_segments(self):
        self.assertEqual(validate_model_version("v11-dusha_62k"), "v11-dusha_62k")
        with self.assertRaises(HTTPException):
            validate_model_version("../outside")


if __name__ == "__main__":
    unittest.main()
