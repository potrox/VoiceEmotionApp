"""Проверки сегментации и подготовки признаков для распознавания."""

from __future__ import annotations

import numpy as np

from app.recognizer import Recognizer, SegmentRecognitionResult


def test_segment_split_drops_only_uninformative_tail() -> None:
    audio_signal = np.ones(1_049, dtype=np.float32)

    segments = Recognizer._split_signal_into_segments(
        audio_signal, sample_rate=100, segment_duration_seconds=5.0
    )

    assert [(start, end) for start, end, _signal in segments] == [
        (0, 500),
        (500, 1_000),
    ]


def test_average_probabilities_uses_every_segment() -> None:
    segment_results = [
        SegmentRecognitionResult(
            1, 0.0, 5.0, "joy", "Радость", 0.8,
            {"joy": 0.8, "calm": 0.2}, []
        ),
        SegmentRecognitionResult(
            2, 5.0, 10.0, "calm", "Спокойствие", 0.6,
            {"joy": 0.4, "calm": 0.6}, []
        ),
    ]

    probabilities = Recognizer._average_probabilities(segment_results)

    assert np.isclose(probabilities["joy"], 0.6)
    assert np.isclose(probabilities["calm"], 0.4)


def test_feature_schema_mismatch_is_not_silently_padded() -> None:
    class ThreeFeatureExtractor:
        def extract_signal(self, _audio_signal, _sample_rate):
            return np.zeros(3, dtype=np.float32), ["one", "two", "three"]

    recognizer = Recognizer.__new__(Recognizer)
    recognizer.extractor = ThreeFeatureExtractor()
    recognizer.bundle = {"feature_names": ["one", "two"]}

    try:
        recognizer._feature_matrix_from_signals(
            [np.ones(100, dtype=np.float32)], 16_000
        )
    except ValueError as exc:
        assert "Переобучите модель" in str(exc)
    else:
        raise AssertionError("Несовместимый вектор признаков должен быть отклонён")
