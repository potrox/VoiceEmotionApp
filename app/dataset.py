"""Чтение, нормализация и проверка CSV-датасетов голосовых эмоций."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .constants import CSV_COLUMN_ALIASES, EMOTION_EN_TO_CANONICAL, EMOTION_RU, EMOTIONS
from .audio import analyze_quality, get_audio_duration, load_audio, preprocess_signal

EXCLUDE_MARK = "__exclude__"
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3"}

USER_DATASET_FIELDNAMES = [
    "file_path", "original_path", "emotion", "speaker_id", "gender", "age_group", "recorded_at",
    "duration", "sample_rate", "source", "text", "language", "quality",
]


@dataclass(frozen=True)
class DatasetLoadResult:

    dataframe: pd.DataFrame
    original_path: str
    mapping: dict[str, str]
    ignored_labels: dict[str, int]
    excluded_rows: int
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def distribution(self) -> dict[str, int]:
        if self.dataframe.empty:
            return {}
        return self.dataframe["emotion"].value_counts().to_dict()

    @property
    def records_count(self) -> int:
        return int(len(self.dataframe))


@dataclass(frozen=True)
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


def _safe_float(value: Any) -> float | None:
    try:
        text = str(value).strip().replace(",", ".")
        if not text or text.lower() in {"nan", "none", "null"}:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _resolve_audio_path(value: Any, base_dir: Path) -> Path | None:
    text = str(value).strip() if value is not None else ""
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _resolve_audio_path_with_user_dataset_fallback(
    value: Any, base_dir: Path
) -> Path | None:
    path = _resolve_audio_path(value, base_dir)
    if path is None or path.exists():
        return path
    parts_lower = [part.lower() for part in path.parts]
    if "user_dataset" in parts_lower:
        user_dataset_index = parts_lower.index("user_dataset")
        relative_tail = Path(*path.parts[user_dataset_index + 1 :])
        local_candidate = (base_dir / relative_tail).resolve()
        if local_candidate.exists():
            return local_candidate
    return path


def _find_original_for_processed(audio_path: Path) -> Path | None:
    if audio_path.parent.name.lower() == "processed":
        original_candidate = (
            audio_path.parent.parent / "original" / audio_path.name
        )
        if original_candidate.exists():
            return original_candidate
    return None


def _duration_for_training_rule(
    row: pd.Series,
    audio_path: Path,
    base_dir: Path,
    sample_rate: int = 16_000,
) -> tuple[float, str]:
    original_from_column = _resolve_audio_path_with_user_dataset_fallback(
        row.get("original_path"), base_dir
    )
    if original_from_column is not None and original_from_column.exists():
        return (
            get_audio_duration(original_from_column, sample_rate=sample_rate),
            f"original_path: {original_from_column}",
        )

    original_sibling = _find_original_for_processed(audio_path)
    if original_sibling is not None:
        return (
            get_audio_duration(original_sibling, sample_rate=sample_rate),
            f"original: {original_sibling}",
        )

    meta_duration = _safe_float(row.get("duration", None))
    if meta_duration is not None and meta_duration > 0:
        return meta_duration, "duration из CSV"

    return get_audio_duration(audio_path, sample_rate=sample_rate), f"файл датасета: {audio_path}"


def detect_columns(dataframe: pd.DataFrame) -> dict[str, str]:
    detected_columns: dict[str, str] = {}
    normalized_names = {
        str(column).strip().lower(): column for column in dataframe.columns
    }
    for logical_name, aliases in CSV_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias.lower() in normalized_names:
                detected_columns[logical_name] = normalized_names[alias.lower()]
                break
    return detected_columns


def read_csv_flexible(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    encodings = ["utf-8", "utf-8-sig", "cp1251"]
    last_exc: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_exc = exc
            continue
        except Exception as exc:
            last_exc = exc
            break
    raise RuntimeError(f"Не удалось прочитать CSV-файл: {last_exc}") from last_exc


def inspect_emotion_labels(path: str | Path) -> list[EmotionLabelInfo]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV-файл не найден: {csv_path}")
    dataframe = read_csv_flexible(csv_path)
    if dataframe.empty:
        raise ValueError("CSV-файл пустой.")
    detected_columns = detect_columns(dataframe)
    missing_columns = [
        column
        for column in ["file_path", "emotion"]
        if column not in detected_columns
    ]
    if missing_columns:
        raise ValueError(
            "В CSV отсутствуют обязательные поля: "
            + ", ".join(missing_columns)
            + ". Требуются путь к аудиофайлу и метка эмоции."
        )
    label_counts = (
        dataframe[detected_columns["emotion"]]
        .fillna("")
        .astype(str)
        .str.strip()
        .value_counts()
        .to_dict()
    )
    return [
        EmotionLabelInfo(
            label=str(label),
            count=int(count),
            suggested=normalize_emotion(str(label)),
        )
        for label, count in label_counts.items()
    ]


class DatasetLoader:

    def __init__(
        self,
        min_train_duration_sec: float = 3.0,
        validate_audio_quality: bool = True,
        sample_rate: int = 16_000,
    ) -> None:
        self.min_train_duration_sec = min_train_duration_sec
        self.validate_audio_quality = validate_audio_quality
        self.sample_rate = int(sample_rate)

    def _resolve_mapping(
        self, raw_emotion: str, mapping: dict[str, str] | None
    ) -> str | None:
        raw_emotion = str(raw_emotion).strip()
        if mapping is not None:
            mapped = mapping.get(raw_emotion)
            if mapped is None:
                mapped = mapping.get(_label_key(raw_emotion))
            if mapped == EXCLUDE_MARK or not mapped:
                return None
            return mapped if mapped in EMOTIONS else None
        return normalize_emotion(raw_emotion)

    def _validate_audio_record(
        self,
        row: pd.Series,
        audio_path: Path,
        csv_directory: Path,
        row_number: int,
    ) -> list[str]:
        duration_seconds, duration_source = _duration_for_training_rule(
            row, audio_path, csv_directory, sample_rate=self.sample_rate
        )
        if duration_seconds < self.min_train_duration_sec:
            raise ValueError(
                f"исходная запись короче {self.min_train_duration_sec:.1f} секунд "
                f"(длительность: {duration_seconds:.2f} сек.; "
                f"источник проверки: {duration_source})"
            )

        audio_signal, sample_rate = load_audio(
            audio_path, sample_rate=self.sample_rate, mono=True
        )
        processed_signal = preprocess_signal(
            audio_signal,
            normalize=True,
            denoise=True,
            trim=True,
        )
        quality = analyze_quality(
            processed_signal, sample_rate, training=True, min_duration_sec=0.1
        )
        if not quality.ok:
            raise ValueError(
                "запись не прошла проверку качества: " + "; ".join(quality.warnings)
            )
        return (
            [
                f"Строка {row_number}: запись длиннее 10 секунд и при "
                "распознавании будет разделяться на фрагменты."
            ]
            if duration_seconds > 10.0
            else []
        )

    @staticmethod
    def _prepare_record(
        row: pd.Series,
        audio_path: Path,
        canonical_emotion: str,
        csv_directory: Path,
    ) -> dict:
        record = row.to_dict()
        record["file_path"] = str(audio_path)
        original_path = _resolve_audio_path_with_user_dataset_fallback(
            record.get("original_path", ""), csv_directory
        )
        if original_path is not None and original_path.exists():
            record["original_path"] = str(original_path)
        record["emotion"] = canonical_emotion
        return record

    @staticmethod
    def _append_class_balance_warnings(
        dataframe: pd.DataFrame, warnings: list[str]
    ) -> None:
        for emotion in EMOTIONS:
            record_count = int((dataframe["emotion"] == emotion).sum())
            if record_count == 0:
                warnings.append(
                    f"В датасете отсутствует класс {emotion} ({EMOTION_RU[emotion]})."
                )
            elif record_count < 10:
                warnings.append(
                    f"Для эмоции {emotion} ({EMOTION_RU[emotion]}) найдено меньше "
                    f"10 записей: {record_count}. Обучение не блокируется, но "
                    "результат может быть нестабильным."
                )

    @staticmethod
    def _normalized_columns(
        source_dataframe: pd.DataFrame, detected_columns: dict[str, str]
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                logical_name: source_dataframe[source_name]
                for logical_name, source_name in detected_columns.items()
            }
        )

    @staticmethod
    def _report_row_progress(
        progress_callback: Callable[[int, str], None] | None,
        row_index: int,
        row_count: int,
    ) -> None:
        if progress_callback is None:
            return
        if row_index not in {0, row_count - 1} and row_index % 5 != 0:
            return
        progress_value = 15 + int(80 * (row_index + 1) / max(row_count, 1))
        progress_callback(
            min(progress_value, 95),
            f"Проверка записей датасета: {row_index + 1} из {row_count}",
        )

    def _load_row(
        self,
        row: pd.Series,
        csv_directory: Path,
        row_number: int,
        emotion_mapping: dict[str, str] | None,
    ) -> tuple[dict | None, str, str | None, list[str]]:
        raw_file_path = str(row.get("file_path", "")).strip()
        raw_emotion = str(row.get("emotion", "")).strip()
        if not raw_file_path:
            raise ValueError("пустой путь к аудиофайлу")
        audio_path = _resolve_audio_path_with_user_dataset_fallback(
            raw_file_path, csv_directory
        )
        if audio_path is None or not audio_path.exists():
            raise ValueError(f"аудиофайл не найден: {raw_file_path}")
        if audio_path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
            raise ValueError(
                f"неподдерживаемый формат аудио: {audio_path.suffix}"
            )

        canonical_emotion = self._resolve_mapping(raw_emotion, emotion_mapping)
        if canonical_emotion is None:
            return None, raw_emotion, None, []
        row_warnings = (
            self._validate_audio_record(
                row, audio_path, csv_directory, row_number
            )
            if self.validate_audio_quality
            else []
        )
        record = self._prepare_record(
            row, audio_path, canonical_emotion, csv_directory
        )
        return record, raw_emotion, canonical_emotion, row_warnings

    def load_csv(
        self,
        path: str | Path,
        emotion_mapping: dict[str, str] | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> DatasetLoadResult:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV-файл не найден: {csv_path}")
        source_dataframe = read_csv_flexible(csv_path)

        if source_dataframe.empty:
            raise ValueError("CSV-файл пустой.")

        detected_columns = detect_columns(source_dataframe)
        missing_columns = [
            column
            for column in ["file_path", "emotion"]
            if column not in detected_columns
        ]
        if missing_columns:
            raise ValueError(
                "В CSV отсутствуют обязательные поля: " + ", ".join(missing_columns) +
                ". Требуются путь к аудиофайлу и метка эмоции."
            )

        normalized_dataframe = self._normalized_columns(
            source_dataframe, detected_columns
        )

        csv_directory = csv_path.parent
        accepted_records: list[dict] = []
        errors: list[str] = []
        warnings: list[str] = []
        resolved_mapping: dict[str, str] = {}
        ignored_labels: dict[str, int] = {}

        row_count = len(normalized_dataframe)
        for row_index, row in normalized_dataframe.iterrows():
            row_number = int(row_index) + 2
            self._report_row_progress(progress_callback, row_index, row_count)
            try:
                record, raw_emotion, canonical_emotion, row_warnings = (
                    self._load_row(
                        row,
                        csv_directory,
                        row_number,
                        emotion_mapping,
                    )
                )
            except Exception as exc:
                errors.append(f"Строка {row_number}: {exc}.")
                continue
            if record is None or canonical_emotion is None:
                ignored_labels[raw_emotion] = ignored_labels.get(raw_emotion, 0) + 1
                continue
            accepted_records.append(record)
            resolved_mapping[raw_emotion] = canonical_emotion
            warnings.extend(row_warnings)

        total_rows = len(source_dataframe)
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

        cleaned_dataframe = pd.DataFrame(accepted_records)
        if cleaned_dataframe.empty:
            raise ValueError("После проверки не осталось пригодных записей. Проверьте пути к файлам, качество аудио и метки эмоций.")

        self._append_class_balance_warnings(cleaned_dataframe, warnings)

        return DatasetLoadResult(
            dataframe=cleaned_dataframe.reset_index(drop=True),
            original_path=str(csv_path),
            mapping=resolved_mapping,
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
    dataset_csv_path = Path(csv_path)
    dataset_csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = dataset_csv_path.exists()
    with dataset_csv_path.open(
        "a", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=USER_DATASET_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
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
            }
        )
