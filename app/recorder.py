from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .audio import analyze_quality, preprocess_signal, record_microphone, save_wav
from .constants import EMOTIONS
from .dataset import create_user_dataset_row
from .storage import timestamp




@dataclass
class PendingRecording:
    samples: object
    cache_path: Path
    quality: object


@dataclass
class SavedRecording:
    original_path: Path
    processed_path: Path
    csv_path: Path
    warnings: list[str]


class UserDatasetRecorder:
    def __init__(self, datasets_dir: str | Path, sample_rate: int = 16000):
        self.datasets_dir = Path(datasets_dir)
        self.sample_rate = sample_rate
        self.base = self.datasets_dir / "user_dataset"
        self.base.mkdir(parents=True, exist_ok=True)

    def next_speaker_id(self) -> str:
        existing = [p.name for p in self.base.glob("speaker_*") if p.is_dir()]
        numbers = []
        for name in existing:
            try:
                numbers.append(int(name.split("_")[-1]))
            except Exception:
                pass
        return f"speaker_{(max(numbers) + 1) if numbers else 1:03d}"

    def record(self, duration_sec: int, input_device_index: Optional[int] = None, cache_dir: str | Path | None = None) -> PendingRecording:
        y = record_microphone(duration_sec, self.sample_rate, device_index=input_device_index)
        quality = analyze_quality(y, self.sample_rate, training=False)
        cache_base = Path(cache_dir) if cache_dir is not None else self.datasets_dir.parent / "cache"
        cache_base.mkdir(parents=True, exist_ok=True)
        cache_path = cache_base / f"microphone_cache_{timestamp()}.wav"
        save_wav(cache_path, y, self.sample_rate)
        return PendingRecording(samples=y, cache_path=cache_path, quality=quality)

    def save_recording(self, y, speaker_id: str, emotion: str, gender: str, age_group: str, phrase: str, language: str = "ru") -> SavedRecording:
        if not speaker_id:
            raise ValueError("Сохранение записи без идентификатора диктора не допускается.")
        if emotion not in EMOTIONS:
            raise ValueError("Сохранение записи без поддерживаемой эмоции не допускается.")
        if not gender or gender == "не указано":
            raise ValueError("Сохранение записи без указания пола диктора не допускается.")
        if not age_group:
            raise ValueError("Сохранение записи без возрастной группы диктора не допускается.")
        if not phrase.strip():
            raise ValueError("Сохранение записи без текста фразы не допускается.")
        quality = analyze_quality(y, self.sample_rate, training=True)
        if not quality.ok:
            raise ValueError("Запись отклонена: " + "; ".join(quality.warnings))
        stamp = timestamp()
        emotion_dir = self.base / speaker_id / emotion
        original_dir = emotion_dir / "original"
        processed_dir = emotion_dir / "processed"
        idx = len(list(original_dir.glob("*.wav"))) + 1
        filename = f"{speaker_id}_{emotion}_{stamp}_{idx:03d}.wav"
        original_path = save_wav(original_dir / filename, y, self.sample_rate)
        y_processed = preprocess_signal(y, self.sample_rate, normalize=True, denoise=True, trim=True)
        processed_path = save_wav(processed_dir / filename, y_processed, self.sample_rate)
        csv_path = self.base / "user_dataset.csv"



        try:
            csv_processed_path = processed_path.relative_to(self.base)
        except Exception:
            csv_processed_path = processed_path
        try:
            csv_original_path = original_path.relative_to(self.base)
        except Exception:
            csv_original_path = original_path
        create_user_dataset_row(
            csv_path=csv_path,
            file_path=csv_processed_path,
            original_path=csv_original_path,
            emotion=emotion,
            speaker_id=speaker_id,
            gender=gender,
            age_group=age_group,
            text=phrase,
            language=language,



            duration=len(y) / self.sample_rate,
            sample_rate=self.sample_rate,
        )
        return SavedRecording(original_path=Path(original_path), processed_path=Path(processed_path), csv_path=csv_path, warnings=quality.warnings)
