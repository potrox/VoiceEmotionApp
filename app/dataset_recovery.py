"""Проверка и восстановление CSV пользовательского голосового датасета."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .audio import get_audio_duration
from .constants import EMOTIONS, LANGUAGES
from .dataset import USER_DATASET_FIELDNAMES, normalize_emotion

AUDIO_EXTENSIONS = {".wav", ".mp3"}
FILENAME_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})")
REQUIRED_RECOVERY_FIELDS = {
    "file_path",
    "emotion",
    "speaker_id",
    "gender",
    "age_group",
    "language",
}


@dataclass(frozen=True)
class RecoveredRecording:

    speaker_id: str
    emotion: str
    file_path: Path
    original_path: Path | None
    duration: float
    sample_rate: int
    recorded_at: str


@dataclass(frozen=True)
class CsvHealth:

    needs_rebuild: bool
    reason: str
    csv_rows: int = 0
    files_found: int = 0


def _safe_relative(path: Path, base_directory: Path) -> str:
    resolved_path = path.resolve()
    resolved_base = base_directory.resolve()
    if resolved_path.is_relative_to(resolved_base):
        return str(resolved_path.relative_to(resolved_base)).replace("\\", "/")
    return str(resolved_path)


def _parse_recorded_at(path: Path) -> str:
    filename_match = FILENAME_DATE_RE.search(path.name)
    if filename_match:
        date_part, time_part = filename_match.groups()
        return f"{date_part}T{time_part.replace('-', ':')}"
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(
            timespec="seconds"
        )
    except OSError:
        return datetime.now().isoformat(timespec="seconds")


def _audio_duration(path: Path, sample_rate: int) -> float:
    try:
        return float(get_audio_duration(path, sample_rate=sample_rate))
    except Exception:
        return 0.0


def _normalize_existing_csv_path(value: str, csv_directory: Path) -> str:
    raw_path = str(value or "").strip()
    if not raw_path:
        return ""
    audio_path = Path(raw_path)
    if not audio_path.is_absolute():
        audio_path = csv_directory / audio_path
    if audio_path.exists():
        return str(audio_path.resolve()).lower()
    normalized_parts = [part.lower() for part in audio_path.parts]
    if "user_dataset" in normalized_parts:
        user_dataset_index = normalized_parts.index("user_dataset")
        relative_tail = Path(*audio_path.parts[user_dataset_index + 1 :])
        audio_path = csv_directory / relative_tail
    return str(audio_path.resolve()).lower()


def _audio_files(directory: Path) -> dict[str, Path]:
    if not directory.exists():
        return {}
    return {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    }


def scan_user_dataset_recordings(
    user_dataset_dir: str | Path, sample_rate: int = 16_000
) -> list[RecoveredRecording]:
    dataset_root = Path(user_dataset_dir)
    if not dataset_root.exists():
        return []
    recordings: list[RecoveredRecording] = []
    seen_recordings: set[tuple[str, str, str]] = set()

    speaker_directories = sorted(
        path for path in dataset_root.glob("speaker_*") if path.is_dir()
    )
    for speaker_directory in speaker_directories:
        speaker_id = speaker_directory.name
        emotion_directories = sorted(
            path for path in speaker_directory.iterdir() if path.is_dir()
        )
        for emotion_directory in emotion_directories:
            emotion = (
                normalize_emotion(emotion_directory.name)
                or emotion_directory.name.lower().strip()
            )
            if emotion not in EMOTIONS:
                continue
            processed_by_name = _audio_files(emotion_directory / "processed")
            original_by_name = _audio_files(emotion_directory / "original")
            for filename in sorted(set(processed_by_name) | set(original_by_name)):
                training_path = processed_by_name.get(filename) or original_by_name.get(
                    filename
                )
                if training_path is None:
                    continue
                recording_key = (speaker_id, emotion, filename.lower())
                if recording_key in seen_recordings:
                    continue
                seen_recordings.add(recording_key)
                original_path = original_by_name.get(filename)
                duration_source = original_path or training_path
                recordings.append(
                    RecoveredRecording(
                        speaker_id=speaker_id,
                        emotion=emotion,
                        file_path=training_path.resolve(),
                        original_path=(
                            original_path.resolve() if original_path else None
                        ),
                        duration=_audio_duration(duration_source, sample_rate),
                        sample_rate=sample_rate,
                        recorded_at=_parse_recorded_at(training_path),
                    )
                )
    return recordings


def _read_existing_csv(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            with csv_path.open("r", encoding=encoding, newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                field_names = list(reader.fieldnames or [])
                rows = []
                for row in reader:
                    if None in row:
                        raise ValueError(
                            "в одной из строк лишние поля или неверное "
                            "экранирование запятых"
                        )
                    rows.append(row)
                return field_names, rows
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except Exception as exc:
            raise ValueError(str(exc)) from exc
    raise ValueError(str(last_error or "неизвестная ошибка чтения"))


def check_user_dataset_csv(
    csv_path: str | Path, records: Iterable[RecoveredRecording]
) -> CsvHealth:
    dataset_csv_path = Path(csv_path)
    recovered_records = list(records)
    file_count = len(recovered_records)
    if not recovered_records:
        return CsvHealth(
            False,
            "В папке пользовательского датасета пока нет аудиозаписей.",
        )
    if not dataset_csv_path.exists():
        return CsvHealth(
            True, "CSV пользовательского датасета отсутствует.", 0, file_count
        )

    try:
        field_names, rows = _read_existing_csv(dataset_csv_path)
    except ValueError as exc:
        return CsvHealth(
            True, f"CSV не удалось прочитать: {exc}", 0, file_count
        )
    missing_fields = sorted(REQUIRED_RECOVERY_FIELDS - set(field_names))
    if missing_fields:
        return CsvHealth(
            True,
            "В CSV отсутствуют служебные поля: " + ", ".join(missing_fields),
            0,
            file_count,
        )
    if len(rows) != file_count:
        return CsvHealth(
            True,
            f"Количество строк CSV ({len(rows)}) не совпадает с количеством "
            f"найденных записей ({file_count}).",
            len(rows),
            file_count,
        )

    expected_files = {
        str(record.file_path.resolve()).lower() for record in recovered_records
    }
    existing_files = {
        _normalize_existing_csv_path(
            row.get("file_path", ""), dataset_csv_path.parent
        )
        for row in rows
    }
    if existing_files != expected_files:
        return CsvHealth(
            True,
            "Пути в CSV не совпадают с текущим расположением аудиофайлов. "
            "Такое часто бывает после копирования датасета с другого компьютера.",
            len(rows),
            file_count,
        )

    incomplete_rows = [
        str(row_number)
        for row_number, row in enumerate(rows, start=2)
        if not str(row.get("gender", "")).strip()
        or str(row.get("gender", "")).strip() == "не указано"
        or not str(row.get("age_group", "")).strip()
        or not str(row.get("language", "")).strip()
    ]
    if incomplete_rows:
        return CsvHealth(
            True,
            "В CSV есть неполные сведения о дикторе в строках: "
            + ", ".join(incomplete_rows[:10]),
            len(rows),
            file_count,
        )
    return CsvHealth(
        False,
        "CSV пользовательского датасета корректен.",
        len(rows),
        file_count,
    )


def _recovery_row(
    record: RecoveredRecording,
    csv_directory: Path,
    metadata: dict[str, str],
    default_text: str,
) -> dict[str, str | int]:
    gender = str(metadata.get("gender", "")).strip()
    age_group = str(metadata.get("age_group", "")).strip()
    language = str(metadata.get("language", "ru")).strip() or "ru"
    if not gender or gender == "не указано":
        raise ValueError(f"Для диктора {record.speaker_id} не указан пол.")
    if not age_group:
        raise ValueError(
            f"Для диктора {record.speaker_id} не указана возрастная группа."
        )
    if language not in LANGUAGES:
        language = "ru"
    return {
        "file_path": _safe_relative(record.file_path, csv_directory),
        "original_path": (
            _safe_relative(record.original_path, csv_directory)
            if record.original_path
            else ""
        ),
        "emotion": record.emotion,
        "speaker_id": record.speaker_id,
        "gender": gender,
        "age_group": age_group,
        "recorded_at": record.recorded_at,
        "duration": f"{record.duration:.3f}",
        "sample_rate": record.sample_rate,
        "source": "microphone",
        "text": str(metadata.get("text", default_text)).strip() or default_text,
        "language": language,
        "quality": "accepted",
    }


def rebuild_user_dataset_csv(
    csv_path: str | Path,
    records: Iterable[RecoveredRecording],
    metadata_by_speaker: dict[str, dict[str, str]],
    default_text: str = "Фраза не указана: CSV восстановлен автоматически",
) -> Path:
    dataset_csv_path = Path(csv_path)
    csv_directory = dataset_csv_path.parent
    csv_directory.mkdir(parents=True, exist_ok=True)
    sorted_records = sorted(
        records,
        key=lambda record: (
            record.speaker_id,
            record.emotion,
            str(record.file_path),
        ),
    )
    rows = [
        _recovery_row(
            record,
            csv_directory,
            metadata_by_speaker.get(record.speaker_id, {}),
            default_text,
        )
        for record in sorted_records
    ]

    backup_path: Path | None = None
    if dataset_csv_path.exists():
        backup_path = dataset_csv_path.with_name(
            f"{dataset_csv_path.stem}_backup_"
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            f"{dataset_csv_path.suffix}"
        )
        try:
            dataset_csv_path.replace(backup_path)
        except OSError:
            backup_path = None

    try:
        with dataset_csv_path.open(
            "w", encoding="utf-8", newline=""
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file, fieldnames=USER_DATASET_FIELDNAMES
            )
            writer.writeheader()
            writer.writerows(rows)
    except Exception:
        if (
            backup_path is not None
            and backup_path.exists()
            and not dataset_csv_path.exists()
        ):
            backup_path.replace(dataset_csv_path)
        raise
    return dataset_csv_path
