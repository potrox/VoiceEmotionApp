"""Проверки восстановления CSV по аудиофайлам."""

from __future__ import annotations

import wave

from app.dataset_recovery import (
    check_user_dataset_csv,
    rebuild_user_dataset_csv,
    scan_user_dataset_recordings,
)


def test_rebuilt_dataset_csv_matches_files_on_disk(tmp_path) -> None:
    dataset_root = tmp_path / "user_dataset"
    original_directory = dataset_root / "speaker_001" / "joy" / "original"
    processed_directory = dataset_root / "speaker_001" / "joy" / "processed"
    original_directory.mkdir(parents=True)
    processed_directory.mkdir(parents=True)
    filename = "speaker_001_joy_2026-01-01_12-00-00_001.wav"
    for audio_path in [
        original_directory / filename,
        processed_directory / filename,
    ]:
        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16_000)
            wav_file.writeframes(b"\x00\x00" * 16_000)

    recordings = scan_user_dataset_recordings(dataset_root)
    csv_path = dataset_root / "user_dataset.csv"
    rebuild_user_dataset_csv(
        csv_path,
        recordings,
        {
            "speaker_001": {
                "gender": "male",
                "age_group": "18-25",
                "language": "ru",
                "text": "Тестовая фраза",
            }
        },
    )

    health = check_user_dataset_csv(csv_path, recordings)

    assert len(recordings) == 1
    assert health.needs_rebuild is False
    assert health.csv_rows == 1
