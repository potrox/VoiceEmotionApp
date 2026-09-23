from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd

from .constants import CSV_COLUMN_ALIASES, EMOTION_EN_TO_CANONICAL, EMOTION_RU, EMOTIONS
from .audio import analyze_quality, get_audio_duration, load_audio, preprocess_signal

logger = logging.getLogger(__name__)

EXCLUDE_MARK = "__exclude__"

USER_DATASET_FIELDNAMES = [
    "file_path", "original_path", "emotion", "speaker_id", "gender", "age_group", "recorded_at",
    "duration", "sample_rate", "source", "text", "language", "quality",
]


@dataclass
class DatasetLoadResult:
    dataframe: pd.DataFrame
    original_path: str
    mapping: Dict[str, str]
    ignored_labels: Dict[str, int]
    excluded_rows: int
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def distribution(self) -> Dict[str, int]:
        if self.dataframe.empty:
            return {}
        return self.dataframe["emotion"].value_counts().to_dict()

    @property
    def records_count(self) -> int:
        return int(len(self.dataframe))


@dataclass
class EmotionLabelInfo:
    label: str
    count: int
    suggested: str | None


def _label_key(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def normalize_emotion(value: str) -> str | None:
    if value is None:
        return None
    return EMOTION_EN_TO_CANONICAL.get(_label_key(value))


def _safe_float(value) -> float | None:
    try:
        text = str(value).strip().replace(",", ".")
        if not text or text.lower() in {"nan", "none", "null"}:
            return None
        return float(text)
    except Exception:
        return None


def _resolve_audio_path(value, base_dir: Path) -> Path | None:
    text = str(value).strip() if value is not None else ""
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _resolve_audio_path_with_user_dataset_fallback(value, base_dir: Path) -> Path | None:
    """Resolve a CSV path and repair copied user-dataset absolute paths when possible.

    Older CSV rows could contain absolute paths from another computer, for example
    C:/Users/.../VoiceEmotionApp/datasets/user_dataset/speaker_001/joy/processed/x.wav.
    After copying only the dataset folder those paths do not exist.  If the tail
    after user_dataset exists in the current dataset directory, use it.
    """
    path = _resolve_audio_path(value, base_dir)
    if path is None or path.exists():
        return path
    parts_lower = [part.lower() for part in path.parts]
    if "user_dataset" in parts_lower:
        try:
            index = parts_lower.index("user_dataset")
            relative_tail = Path(*path.parts[index + 1:])
            candidate = (base_dir / relative_tail).resolve()
            if candidate.exists():
                return candidate
        except Exception:
            pass
    return path


def _find_original_for_processed(audio_path: Path) -> Path | None:
    """Ищет original-версию для файла из папки processed пользовательского датасета."""
    try:
        if audio_path.parent.name.lower() == "processed":
            candidate = audio_path.parent.parent / "original" / audio_path.name
            if candidate.exists():
                return candidate
    except Exception:
        return None
    return None


def _duration_for_training_rule(row, audio_path: Path, base_dir: Path, sample_rate: int = 16000) -> tuple[float, str]:
    """Определяет длительность именно исходной записи для правила 3 секунд."""
    original_from_column = _resolve_audio_path_with_user_dataset_fallback(row.get("original_path", None), base_dir)
    if original_from_column is not None and original_from_column.exists():
        return get_audio_duration(original_from_column, sample_rate=sample_rate), f"original_path: {original_from_column}"

    original_sibling = _find_original_for_processed(audio_path)
    if original_sibling is not None:
        return get_audio_duration(original_sibling, sample_rate=sample_rate), f"original: {original_sibling}"

    meta_duration = _safe_float(row.get("duration", None))
    if meta_duration is not None and meta_duration > 0:
        return meta_duration, "duration из CSV"

    return get_audio_duration(audio_path, sample_rate=sample_rate), f"файл датасета: {audio_path}"


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    detected: Dict[str, str] = {}
    lower_to_real = {str(col).strip().lower(): col for col in df.columns}
    for logical, aliases in CSV_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lower_to_real:
                detected[logical] = lower_to_real[alias.lower()]
                break
    return detected


def read_csv_flexible(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    encodings = ["utf-8", "utf-8-sig", "cp1251"]
    last_exc: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_exc = exc
            continue
        except Exception as exc:
            last_exc = exc
            break
    raise RuntimeError(f"Не удалось прочитать CSV-файл: {last_exc}") from last_exc


def inspect_emotion_labels(path: str | Path) -> List[EmotionLabelInfo]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV-файл не найден: {path}")
    df = read_csv_flexible(path)
    if df.empty:
        raise ValueError("CSV-файл пустой.")
    detected = detect_columns(df)
    missing = [col for col in ["file_path", "emotion"] if col not in detected]
    if missing:
        raise ValueError(
            "В CSV отсутствуют обязательные поля: " + ", ".join(missing) +
            ". Требуются путь к аудиофайлу и метка эмоции."
        )
    counts = df[detected["emotion"]].fillna("").astype(str).str.strip().value_counts().to_dict()
    return [EmotionLabelInfo(label=str(label), count=int(count), suggested=normalize_emotion(str(label))) for label, count in counts.items()]


class DatasetLoader:
    def __init__(self, min_train_duration_sec: float = 3.0, validate_audio_quality: bool = True):
        self.min_train_duration_sec = min_train_duration_sec
        self.validate_audio_quality = validate_audio_quality

    def _resolve_mapping(self, raw_emotion: str, mapping: Dict[str, str] | None) -> str | None:
        raw_emotion = str(raw_emotion).strip()
        if mapping is not None:
            mapped = mapping.get(raw_emotion)
            if mapped is None:
                mapped = mapping.get(_label_key(raw_emotion))
            if mapped == EXCLUDE_MARK or not mapped:
                return None
            return mapped if mapped in EMOTIONS else None
        return normalize_emotion(raw_emotion)

    def load_csv(self, path: str | Path, emotion_mapping: Dict[str, str] | None = None, progress_callback: Callable[[int, str], None] | None = None) -> DatasetLoadResult:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV-файл не найден: {path}")
        df = read_csv_flexible(path)

        if df.empty:
            raise ValueError("CSV-файл пустой.")

        detected = detect_columns(df)
        missing = [col for col in ["file_path", "emotion"] if col not in detected]
        if missing:
            raise ValueError(
                "В CSV отсутствуют обязательные поля: " + ", ".join(missing) +
                ". Требуются путь к аудиофайлу и метка эмоции."
            )

        working = pd.DataFrame()
        for logical, real in detected.items():
            working[logical] = df[real]

        base_dir = path.parent
        rows = []
        errors: List[str] = []
        warnings: List[str] = []
        mapping: Dict[str, str] = {}
        ignored_labels: Dict[str, int] = {}

        total_iter_rows = max(len(working), 1)
        for idx, row in working.iterrows():
            if progress_callback and (idx == 0 or idx % 5 == 0 or idx == len(working) - 1):
                value = 15 + int(80 * (idx + 1) / total_iter_rows)
                progress_callback(min(value, 95), f"Проверка записей датасета: {idx + 1} из {len(working)}")
            raw_file = str(row.get("file_path", "")).strip()
            raw_emotion = str(row.get("emotion", "")).strip()
            if not raw_file:
                errors.append(f"Строка {idx + 2}: пустой путь к аудиофайлу.")
                continue
            audio_path = _resolve_audio_path_with_user_dataset_fallback(raw_file, base_dir)
            if audio_path is None or not audio_path.exists():
                errors.append(f"Строка {idx + 2}: аудиофайл не найден: {audio_path}")
                continue
            if audio_path.suffix.lower() not in [".wav", ".mp3"]:
                errors.append(f"Строка {idx + 2}: неподдерживаемый формат аудио: {audio_path.suffix}")
                continue

            canonical = self._resolve_mapping(raw_emotion, emotion_mapping)
            if canonical is None:
                ignored_labels[raw_emotion] = ignored_labels.get(raw_emotion, 0) + 1
                continue
            mapping[raw_emotion] = canonical

            if self.validate_audio_quality:
                try:
                    duration_for_rule, duration_source = _duration_for_training_rule(row, audio_path, base_dir, sample_rate=16000)

                    if duration_for_rule < self.min_train_duration_sec:
                        errors.append(
                            f"Строка {idx + 2}: исходная запись короче {self.min_train_duration_sec:.1f} секунд "
                            f"(длительность: {duration_for_rule:.2f} сек.; источник проверки: {duration_source})."
                        )
                        continue

                    y_raw, sr = load_audio(audio_path, sample_rate=16000, mono=True)
                    y_processed = preprocess_signal(y_raw, sr, normalize=True, denoise=True, trim=True)
                    quality = analyze_quality(y_processed, sr, training=True, min_duration_sec=0.1)
                    if not quality.ok:
                        errors.append(f"Строка {idx + 2}: запись не прошла проверку качества: {'; '.join(quality.warnings)}")
                        continue
                    if duration_for_rule > 10.0:
                        warnings.append(f"Строка {idx + 2}: запись длиннее 10 секунд и при распознавании будет разделяться на фрагменты.")
                except Exception as exc:
                    errors.append(f"Строка {idx + 2}: не удалось проверить аудио: {exc}")
                    continue

            record = row.to_dict()
            record["file_path"] = str(audio_path)
            original_path = _resolve_audio_path_with_user_dataset_fallback(record.get("original_path", ""), base_dir)
            if original_path is not None and original_path.exists():
                record["original_path"] = str(original_path)
            record["emotion"] = canonical
            rows.append(record)

        total_rows = len(df)
        error_ratio = len(errors) / max(total_rows, 1)
        if error_ratio > 0.05:
            raise ValueError(
                f"В датасете слишком много ошибочных записей: {len(errors)} из {total_rows} "
                f"({error_ratio:.1%}). Обучение остановлено до исправления датасета.\n" + "\n".join(errors[:30])
            )
        if errors:
            warnings.append(f"Обнаружены ошибочные записи: {len(errors)} из {total_rows}. Так как их меньше 5%, работа может быть продолжена.")

        if ignored_labels:
            warnings.append("Исключены неподдерживаемые эмоции: " + ", ".join(f"{k} ({v})" for k, v in ignored_labels.items()))

        cleaned = pd.DataFrame(rows)
        if cleaned.empty:
            raise ValueError("После проверки не осталось пригодных записей. Проверьте пути к файлам, качество аудио и метки эмоций.")

        for emotion in EMOTIONS:
            count = int((cleaned["emotion"] == emotion).sum())
            if count == 0:
                warnings.append(f"В датасете отсутствует класс {emotion} ({EMOTION_RU[emotion]}).")
            elif count < 10:
                warnings.append(f"Для эмоции {emotion} ({EMOTION_RU[emotion]}) найдено меньше 10 записей: {count}. Обучение не блокируется, но результат может быть нестабильным.")

        return DatasetLoadResult(
            dataframe=cleaned.reset_index(drop=True),
            original_path=str(path),
            mapping=mapping,
            ignored_labels=ignored_labels,
            excluded_rows=len(errors) + sum(ignored_labels.values()),
            warnings=warnings,
            errors=errors,
        )


def create_user_dataset_row(
    csv_path: str | Path,
    file_path: str | Path,
    emotion: str,
    speaker_id: str,
    gender: str,
    age_group: str,
    text: str,
    language: str,
    duration: float,
    sample_rate: int,
    original_path: str | Path | None = None,
    source: str = "microphone",
    quality: str = "accepted",
) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    from datetime import datetime
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=USER_DATASET_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "file_path": str(file_path),
            "original_path": str(original_path) if original_path else "",
            "emotion": emotion,
            "speaker_id": speaker_id,
            "gender": gender,
            "age_group": age_group,
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "duration": f"{duration:.3f}",
            "sample_rate": sample_rate,
            "source": source,
            "text": text,
            "language": language,
            "quality": quality,
        })
