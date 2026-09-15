"""Проверки загрузки и нормализации датасета."""

from __future__ import annotations

import pandas as pd

from app.dataset import DatasetLoader


def test_loader_normalizes_aliases_and_relative_paths(tmp_path) -> None:
    first_audio = tmp_path / "first.wav"
    second_audio = tmp_path / "second.mp3"
    first_audio.write_bytes(b"wav")
    second_audio.write_bytes(b"mp3")
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        [
            {"path": first_audio.name, "label": "joy", "speaker": "one"},
            {"path": second_audio.name, "label": "calm", "speaker": "two"},
        ]
    ).to_csv(csv_path, index=False)

    result = DatasetLoader(validate_audio_quality=False).load_csv(csv_path)

    assert result.records_count == 2
    assert result.distribution == {"joy": 1, "calm": 1}
    assert all(
        path.startswith(str(tmp_path)) for path in result.dataframe["file_path"]
    )
    assert result.dataframe["speaker_id"].tolist() == ["one", "two"]
