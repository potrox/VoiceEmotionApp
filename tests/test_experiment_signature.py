"""Проверки подписи обучающего эксперимента."""

from __future__ import annotations

import os

import pandas as pd

from app.config import AppConfig
from app.dataset import DatasetLoadResult
from app.services.experiment_signature import build_experiment_signature


def test_signature_is_order_independent_and_manual_parameters_are_explicit(tmp_path) -> None:
    first_audio = tmp_path / "first.wav"
    second_audio = tmp_path / "second.wav"
    first_audio.write_bytes(b"first")
    second_audio.write_bytes(b"second")
    rows = [
        {"file_path": str(first_audio), "emotion": "calm", "speaker_id": "one"},
        {"file_path": str(second_audio), "emotion": "joy", "speaker_id": "two"},
    ]
    config = AppConfig(random_state=42)

    def dataset(dataframe: pd.DataFrame) -> DatasetLoadResult:
        return DatasetLoadResult(dataframe, "dataset.csv", {}, {}, 0)

    first_signature = build_experiment_signature(
        dataset(pd.DataFrame(rows)), config, ["SVM", "MLP"], "auto", {"ignored": 1}
    )
    reordered_signature = build_experiment_signature(
        dataset(pd.DataFrame(list(reversed(rows)))), config, ["MLP", "SVM"], "auto", {}
    )
    manual_signature = build_experiment_signature(
        dataset(pd.DataFrame(rows)), config, ["SVM", "MLP"], "manual", {"svm": {"C": 5}}
    )
    original_metadata = first_audio.stat()
    first_audio.write_bytes(b"other")
    os.utime(
        first_audio,
        ns=(original_metadata.st_atime_ns, original_metadata.st_mtime_ns),
    )
    changed_content_signature = build_experiment_signature(
        dataset(pd.DataFrame(rows)), config, ["SVM", "MLP"], "auto", {}
    )

    assert first_signature == reordered_signature
    assert manual_signature != first_signature
    assert changed_content_signature != first_signature
