from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .audio import get_audio_duration
from .constants import EMOTIONS, GENDERS, LANGUAGES
from .dataset import USER_DATASET_FIELDNAMES, normalize_emotion

AUDIO_EXTENSIONS = {".wav", ".mp3"}
FILENAME_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})")


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


def _safe_relative(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def _parse_recorded_at(path: Path) -> str:
    match = FILENAME_DATE_RE.search(path.name)
    if match:
        date_part, time_part = match.groups()
        return f"{date_part}T{time_part.replace('-', ':')}"
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except Exception:
        return datetime.now().isoformat(timespec="seconds")


def _duration(path: Path, sample_rate: int) -> float:
    try:
        return float(get_audio_duration(path, sample_rate=sample_rate))
    except Exception:
        return 0.0


def _normalize_existing_csv_path(value: str, csv_base: Path) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = csv_base / path
    if path.exists():
        return str(path.resolve()).lower()
    parts_lower = [part.lower() for part in path.parts]
    if "user_dataset" in parts_lower:
        try:
            index = parts_lower.index("user_dataset")
            candidate = csv_base / Path(*path.parts[index + 1:])
            return str(candidate.resolve()).lower()
        except Exception:
            pass
    try:
        return str(path.resolve()).lower()
    except Exception:
        return str(path).lower()


def scan_user_dataset_recordings(user_dataset_dir: str | Path, sample_rate: int = 16000) -> List[RecoveredRecording]:
    """Scan datasets/user_dataset and return one row per available recording.

    The preferred training file is the processed version.  If only original files
    are present, the original file is used as file_path so a copied dataset can
    still be restored instead of losing records.
    """
    base = Path(user_dataset_dir)
    if not base.exists():
        return []
    records: list[RecoveredRecording] = []
    seen: set[Tuple[str, str, str]] = set()

    for speaker_dir in sorted(p for p in base.glob("speaker_*") if p.is_dir()):
        speaker_id = speaker_dir.name
        for emotion_dir in sorted(p for p in speaker_dir.iterdir() if p.is_dir()):
            emotion = normalize_emotion(emotion_dir.name) or emotion_dir.name.lower().strip()
            if emotion not in EMOTIONS:
                continue
            processed_dir = emotion_dir / "processed"
            original_dir = emotion_dir / "original"

            processed_files = [p for p in processed_dir.glob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS] if processed_dir.exists() else []
            original_files = [p for p in original_dir.glob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS] if original_dir.exists() else []

            processed_by_name = {p.name: p for p in processed_files}
            original_by_name = {p.name: p for p in original_files}
            all_names = sorted(set(processed_by_name) | set(original_by_name))

            for name in all_names:
                training_path = processed_by_name.get(name) or original_by_name.get(name)
                original_path = original_by_name.get(name)
                if training_path is None:
                    continue
                key = (speaker_id, emotion, name.lower())
                if key in seen:
                    continue
                seen.add(key)
                source_for_duration = original_path or training_path
                records.append(RecoveredRecording(
                    speaker_id=speaker_id,
                    emotion=emotion,
                    file_path=training_path.resolve(),
                    original_path=original_path.resolve() if original_path else None,
                    duration=_duration(source_for_duration, sample_rate),
                    sample_rate=sample_rate,
                    recorded_at=_parse_recorded_at(training_path),
                ))
    return records


def check_user_dataset_csv(csv_path: str | Path, records: Iterable[RecoveredRecording]) -> CsvHealth:
    csv_path = Path(csv_path)
    record_list = list(records)
    expected_files = {str(record.file_path.resolve()).lower() for record in record_list}
    if not record_list:
        return CsvHealth(False, "В папке пользовательского датасета пока нет аудиозаписей.", 0, 0)
    if not csv_path.exists():
        return CsvHealth(True, "CSV пользовательского датасета отсутствует.", 0, len(record_list))

    rows = []
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            required = {"file_path", "emotion", "speaker_id", "gender", "age_group", "language"}
            missing = sorted(required - set(fieldnames))
            if missing:
                return CsvHealth(True, "В CSV отсутствуют служебные поля: " + ", ".join(missing), 0, len(record_list))
            for row in reader:
                if None in row:
                    return CsvHealth(True, "CSV повреждён: в одной из строк лишние поля или неверное экранирование запятых.", len(rows), len(record_list))
                rows.append(row)
    except UnicodeDecodeError:
        try:
            with csv_path.open("r", encoding="cp1251", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception as exc:
            return CsvHealth(True, f"CSV не удалось прочитать: {exc}", 0, len(record_list))
    except Exception as exc:
        return CsvHealth(True, f"CSV не удалось прочитать: {exc}", 0, len(record_list))

    if len(rows) != len(record_list):
        return CsvHealth(True, f"Количество строк CSV ({len(rows)}) не совпадает с количеством найденных записей ({len(record_list)}).", len(rows), len(record_list))

    existing_files = {_normalize_existing_csv_path(row.get("file_path", ""), csv_path.parent) for row in rows}
    if existing_files != expected_files:
        return CsvHealth(True, "Пути в CSV не совпадают с текущим расположением аудиофайлов. Такое часто бывает после копирования датасета с другого компьютера.", len(rows), len(record_list))

    invalid_meta = []
    for number, row in enumerate(rows, start=2):
        gender = str(row.get("gender", "")).strip()
        age_group = str(row.get("age_group", "")).strip()
        language = str(row.get("language", "")).strip()
        if not gender or gender == "не указано" or not age_group or not language:
            invalid_meta.append(str(number))
    if invalid_meta:
        return CsvHealth(True, "В CSV есть неполные сведения о дикторе в строках: " + ", ".join(invalid_meta[:10]), len(rows), len(record_list))

    return CsvHealth(False, "CSV пользовательского датасета корректен.", len(rows), len(record_list))


def rebuild_user_dataset_csv(
    csv_path: str | Path,
    records: Iterable[RecoveredRecording],
    metadata_by_speaker: Dict[str, Dict[str, str]],
    default_text: str = "Фраза не указана: CSV восстановлен автоматически",
) -> Path:
    csv_path = Path(csv_path)
    base = csv_path.parent
    base.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in sorted(records, key=lambda r: (r.speaker_id, r.emotion, str(r.file_path))):
        meta = metadata_by_speaker.get(record.speaker_id, {})
        gender = str(meta.get("gender", "")).strip()
        age_group = str(meta.get("age_group", "")).strip()
        language = str(meta.get("language", "ru")).strip() or "ru"
        if not gender or gender == "не указано":
            raise ValueError(f"Для диктора {record.speaker_id} не указан пол.")
        if not age_group:
            raise ValueError(f"Для диктора {record.speaker_id} не указана возрастная группа.")
        if language not in LANGUAGES:
            language = "ru"
        rows.append({
            "file_path": _safe_relative(record.file_path, base),
            "original_path": _safe_relative(record.original_path, base) if record.original_path else "",
            "emotion": record.emotion,
            "speaker_id": record.speaker_id,
            "gender": gender,
            "age_group": age_group,
            "recorded_at": record.recorded_at,
            "duration": f"{record.duration:.3f}",
            "sample_rate": record.sample_rate,
            "source": "microphone",
            "text": str(meta.get("text", default_text)).strip() or default_text,
            "language": language,
            "quality": "accepted",
        })

    backup_path = None
    if csv_path.exists():
        backup_path = csv_path.with_name(f"{csv_path.stem}_backup_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}{csv_path.suffix}")
        try:
            csv_path.replace(backup_path)
        except Exception:
            backup_path = None

    try:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=USER_DATASET_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
    except Exception:
        if backup_path is not None and backup_path.exists() and not csv_path.exists():
            backup_path.replace(csv_path)
        raise
    return csv_path
