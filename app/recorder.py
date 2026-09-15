"""Сохранение размеченных записей в пользовательский датасет."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import analyze_quality, preprocess_signal, save_wav
from .constants import EMOTIONS
from .dataset import create_user_dataset_row
from .storage import timestamp


@dataclass(frozen=True)
class SavedRecording:

    original_path: Path
    processed_path: Path
    csv_path: Path
    warnings: list[str]


class UserDatasetRecorder:

    def __init__(self, datasets_dir: str | Path, sample_rate: int = 16_000) -> None:
        self.datasets_dir = Path(datasets_dir)
        self.sample_rate = int(sample_rate)
        self.dataset_root = self.datasets_dir / "user_dataset"
        self.dataset_root.mkdir(parents=True, exist_ok=True)

    def next_speaker_id(self) -> str:
        speaker_numbers: list[int] = []
        for speaker_directory in self.dataset_root.glob("speaker_*"):
            if not speaker_directory.is_dir():
                continue
            try:
                speaker_numbers.append(
                    int(speaker_directory.name.removeprefix("speaker_"))
                )
            except ValueError:
                continue
        next_number = max(speaker_numbers, default=0) + 1
        return f"speaker_{next_number:03d}"

    def save_recording(
        self,
        audio_signal: np.ndarray,
        speaker_id: str,
        emotion: str,
        gender: str,
        age_group: str,
        phrase: str,
        language: str = "ru",
    ) -> SavedRecording:
        if not speaker_id:
            raise ValueError(
                "Сохранение записи без идентификатора диктора не допускается."
            )
        if emotion not in EMOTIONS:
            raise ValueError(
                "Сохранение записи без поддерживаемой эмоции не допускается."
            )
        if not gender or gender == "не указано":
            raise ValueError(
                "Сохранение записи без указания пола диктора не допускается."
            )
        if not age_group:
            raise ValueError(
                "Сохранение записи без возрастной группы диктора не допускается."
            )
        if not phrase.strip():
            raise ValueError("Сохранение записи без текста фразы не допускается.")

        quality = analyze_quality(audio_signal, self.sample_rate, training=True)
        if not quality.ok:
            raise ValueError("Запись отклонена: " + "; ".join(quality.warnings))

        emotion_directory = self.dataset_root / speaker_id / emotion
        original_directory = emotion_directory / "original"
        processed_directory = emotion_directory / "processed"
        recording_index = len(list(original_directory.glob("*.wav"))) + 1
        filename = (
            f"{speaker_id}_{emotion}_{timestamp()}_{recording_index:03d}.wav"
        )
        original_path = save_wav(
            original_directory / filename, audio_signal, self.sample_rate
        )
        processed_signal = preprocess_signal(
            audio_signal, normalize=True, denoise=True, trim=True
        )
        processed_path = save_wav(
            processed_directory / filename, processed_signal, self.sample_rate
        )
        csv_path = self.dataset_root / "user_dataset.csv"
        create_user_dataset_row(
            csv_path=csv_path,
            file_path=processed_path.relative_to(self.dataset_root),
            original_path=original_path.relative_to(self.dataset_root),
            emotion=emotion,
            speaker_id=speaker_id,
            gender=gender,
            age_group=age_group,
            text=phrase,
            language=language,
            duration=len(audio_signal) / self.sample_rate,
            sample_rate=self.sample_rate,
        )
        return SavedRecording(
            original_path=original_path,
            processed_path=processed_path,
            csv_path=csv_path,
            warnings=quality.warnings,
        )
