"""Журналирование, история SQLite и текстовые отчёты экспериментов."""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import ProjectPaths


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]


def setup_logging(paths: ProjectPaths) -> None:
    paths.ensure()
    log_file = paths.logs / f"voice_emotion_{timestamp()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


@dataclass(frozen=True)
class TrainingReport:

    dataset_path: str
    processed_files: int
    class_distribution: dict[str, int]
    feature_info: dict[str, Any]
    model_results: dict[str, Any]
    best_model_name: str
    warnings: list[str]
    duration_sec: float


class ExperimentDB:

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize_database(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    dataset_path TEXT,
                    selected_models TEXT,
                    parameters_json TEXT,
                    metrics_json TEXT,
                    best_model_name TEXT,
                    model_path TEXT,
                    report_path TEXT,
                    duration_sec REAL,
                    warnings_json TEXT,
                    errors_json TEXT,
                    experiment_signature TEXT
                )
                """
            )
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(experiments)"
                ).fetchall()
            }
            if "experiment_signature" not in columns:
                connection.execute(
                    "ALTER TABLE experiments ADD COLUMN experiment_signature TEXT"
                )
            if "best_model_name" not in columns:
                connection.execute(
                    "ALTER TABLE experiments ADD COLUMN best_model_name TEXT"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiments_signature "
                "ON experiments(experiment_signature)"
            )

    def add_experiment(
        self,
        dataset_path: str,
        selected_models: Iterable[str],
        parameters: dict[str, Any],
        metrics: dict[str, Any],
        best_model_name: str,
        model_path: str,
        report_path: str,
        duration_sec: float,
        warnings: Iterable[str] = (),
        errors: Iterable[str] = (),
        experiment_signature: str | None = None,
    ) -> int:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO experiments
                (created_at, dataset_path, selected_models, parameters_json,
                 metrics_json, best_model_name, model_path, report_path, duration_sec,
                 warnings_json, errors_json, experiment_signature)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(timespec="seconds"),
                    dataset_path,
                    ", ".join(selected_models),
                    json.dumps(parameters, ensure_ascii=False, default=str),
                    json.dumps(metrics, ensure_ascii=False, default=str),
                    best_model_name,
                    model_path,
                    report_path,
                    duration_sec,
                    json.dumps(list(warnings), ensure_ascii=False),
                    json.dumps(list(errors), ensure_ascii=False),
                    experiment_signature,
                ),
            )
            return int(cursor.lastrowid)

    def last_experiments(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connection() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def find_by_signature(
        self, experiment_signature: str
    ) -> dict[str, Any] | None:
        if not experiment_signature:
            return None
        with self._connection() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM experiments WHERE experiment_signature = ? "
                "ORDER BY id DESC LIMIT 1",
                (experiment_signature,),
            ).fetchone()
        return dict(row) if row else None

    def export_csv(self, output_path: str | Path) -> Path:
        rows = self.last_experiments(100_000)
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            destination.write_text("", encoding="utf-8")
            return destination
        with destination.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return destination

    def export_txt(self, output_path: str | Path) -> Path:
        lines: list[str] = []
        for row in self.last_experiments(100_000):
            lines.extend(
                [
                    f"Эксперимент #{row['id']} | {row['created_at']}",
                    f"Датасет: {row['dataset_path']}",
                    f"Модели: {row['selected_models']}",
                    f"Лучшая модель: {row.get('best_model_name') or '—'}",
                    f"Модель: {row['model_path']}",
                    f"Отчёт: {row['report_path']}",
                    f"Длительность: {row['duration_sec']} сек.",
                    "",
                ]
            )
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(lines), encoding="utf-8")
        return destination


def _append_split_section(lines: list[str], split_info: dict[str, Any]) -> None:
    if not split_info:
        return
    lines.extend(
        [
            "",
            "Схема разделения выборки:",
            f"  - Стратегия: {split_info.get('strategy')}",
            f"  - Обучающая выборка: {split_info.get('train_count')}",
            f"  - Тестовая выборка: {split_info.get('test_count')}",
        ]
    )
    report_fields = {
        "Распределение train по эмоциям": "train_class_distribution",
        "Распределение test по эмоциям": "test_class_distribution",
        "Распределение train по дикторам": "train_speaker_distribution",
        "Распределение test по дикторам": "test_speaker_distribution",
    }
    for label, field_name in report_fields.items():
        lines.append(f"  - {label}:")
        lines.append(
            json.dumps(
                split_info.get(field_name, {}),
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )


def _append_model_section(
    lines: list[str], model_name: str, result: dict[str, Any]
) -> None:
    lines.extend(["-" * 60, f"Модель: {model_name}"])
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "cv_mean",
        "cv_std",
        "train_macro_f1",
        "overfit_gap",
        "avg_inference_time_sec",
    ]
    for metric_name in metric_names:
        if metric_name in result:
            lines.append(f"{metric_name}: {result[metric_name]}")
    if "best_params" in result:
        lines.extend(
            [
                "Параметры:",
                json.dumps(
                    result["best_params"],
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
            ]
        )
    if "classification_report" in result:
        lines.extend(
            ["Classification report:", str(result["classification_report"])]
        )
    if "confusion_matrix" in result:
        lines.extend(["Матрица ошибок:", str(result["confusion_matrix"])])
    model_warnings = result.get("warnings", [])
    if model_warnings:
        lines.append("Предупреждения модели:")
        lines.extend(f"  - {warning}" for warning in model_warnings)


def write_training_report(
    report: TrainingReport, output_directory: str | Path
) -> Path:
    report_directory = Path(output_directory)
    report_directory.mkdir(parents=True, exist_ok=True)
    report_path = report_directory / f"training_report_{timestamp()}.txt"
    lines = [
        "ОТЧЁТ ОБ ОБУЧЕНИИ VoiceEmotionApp",
        "=" * 60,
        f"Дата и время обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Путь к датасету: {report.dataset_path}",
        f"Количество обработанных файлов: {report.processed_files}",
        "Распределение записей по эмоциям:",
    ]
    lines.extend(
        f"  - {emotion}: {count}"
        for emotion, count in report.class_distribution.items()
    )
    _append_split_section(lines, report.feature_info.get("split_info") or {})
    lines.extend(
        [
            "",
            "Сведения о признаках:",
            json.dumps(
                report.feature_info,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            "",
            "Результаты моделей:",
        ]
    )
    for model_name, result in report.model_results.items():
        _append_model_section(lines, model_name, result)
    lines.extend(
        [
            "=" * 60,
            f"Выбранная лучшая модель: {report.best_model_name}",
            f"Длительность обучения: {report.duration_sec:.2f} сек.",
        ]
    )
    if report.warnings:
        lines.append("Предупреждения:")
        lines.extend(f"  - {warning}" for warning in report.warnings)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
