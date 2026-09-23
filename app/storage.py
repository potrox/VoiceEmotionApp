from __future__ import annotations

import csv
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .config import ProjectPaths


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def setup_logging(paths: ProjectPaths) -> None:
    paths.ensure()
    log_file = paths.logs / f"voice_emotion_{timestamp()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )


@dataclass
class TrainingReport:
    dataset_path: str
    processed_files: int
    class_distribution: Dict[str, int]
    feature_info: Dict[str, Any]
    model_results: Dict[str, Any]
    best_model_name: str
    warnings: List[str]
    duration_sec: float


class ExperimentDB:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    dataset_path TEXT,
                    selected_models TEXT,
                    parameters_json TEXT,
                    metrics_json TEXT,
                    model_path TEXT,
                    report_path TEXT,
                    duration_sec REAL,
                    warnings_json TEXT,
                    errors_json TEXT,
                    experiment_signature TEXT
                )
                """
            )
            columns = {row[1] for row in con.execute("PRAGMA table_info(experiments)").fetchall()}
            if "experiment_signature" not in columns:
                con.execute("ALTER TABLE experiments ADD COLUMN experiment_signature TEXT")
            con.execute("CREATE INDEX IF NOT EXISTS idx_experiments_signature ON experiments(experiment_signature)")
            con.commit()

    def add_experiment(
        self,
        dataset_path: str,
        selected_models: Iterable[str],
        parameters: Dict[str, Any],
        metrics: Dict[str, Any],
        model_path: str,
        report_path: str,
        duration_sec: float,
        warnings: Iterable[str] = (),
        errors: Iterable[str] = (),
        experiment_signature: str | None = None,
    ) -> int:
        with self._connect() as con:
            cur = con.execute(
                """
                INSERT INTO experiments
                (created_at, dataset_path, selected_models, parameters_json, metrics_json, model_path, report_path, duration_sec, warnings_json, errors_json, experiment_signature)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(timespec="seconds"),
                    dataset_path,
                    ", ".join(selected_models),
                    json.dumps(parameters, ensure_ascii=False, default=str),
                    json.dumps(metrics, ensure_ascii=False, default=str),
                    model_path,
                    report_path,
                    duration_sec,
                    json.dumps(list(warnings), ensure_ascii=False),
                    json.dumps(list(errors), ensure_ascii=False),
                    experiment_signature,
                ),
            )
            con.commit()
            return int(cur.lastrowid)

    def last_experiments(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._connect() as con:
            con.row_factory = sqlite3.Row
            rows = con.execute("SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def find_by_signature(self, experiment_signature: str) -> Dict[str, Any] | None:
        if not experiment_signature:
            return None
        with self._connect() as con:
            con.row_factory = sqlite3.Row
            row = con.execute(
                "SELECT * FROM experiments WHERE experiment_signature = ? ORDER BY id DESC LIMIT 1",
                (experiment_signature,),
            ).fetchone()
        return dict(row) if row else None

    def export_csv(self, out_path: str | Path) -> Path:
        rows = self.last_experiments(100000)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            out_path.write_text("", encoding="utf-8")
            return out_path
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return out_path

    def export_txt(self, out_path: str | Path) -> Path:
        rows = self.last_experiments(100000)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for row in rows:
            lines.append(f"Эксперимент #{row['id']} | {row['created_at']}")
            lines.append(f"Датасет: {row['dataset_path']}")
            lines.append(f"Модели: {row['selected_models']}")
            lines.append(f"Модель: {row['model_path']}")
            lines.append(f"Отчёт: {row['report_path']}")
            lines.append(f"Длительность: {row['duration_sec']} сек.")
            lines.append("")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path


def write_training_report(report: TrainingReport, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"training_report_{timestamp()}.txt"
    lines: List[str] = []
    lines.append("ОТЧЁТ ОБ ОБУЧЕНИИ VoiceEmotionApp")
    lines.append("=" * 60)
    lines.append(f"Дата и время обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Путь к датасету: {report.dataset_path}")
    lines.append(f"Количество обработанных файлов: {report.processed_files}")
    lines.append("Распределение записей по эмоциям:")
    for emotion, count in report.class_distribution.items():
        lines.append(f"  - {emotion}: {count}")
    split_info = report.feature_info.get("split_info") or {}
    if split_info:
        lines.append("")
        lines.append("Схема разделения выборки:")
        lines.append(f"  - Стратегия: {split_info.get('strategy')}")
        lines.append(f"  - Обучающая выборка: {split_info.get('train_count')}")
        lines.append(f"  - Тестовая выборка: {split_info.get('test_count')}")
        lines.append("  - Распределение train по эмоциям:")
        lines.append(json.dumps(split_info.get("train_class_distribution", {}), ensure_ascii=False, indent=2, default=str))
        lines.append("  - Распределение test по эмоциям:")
        lines.append(json.dumps(split_info.get("test_class_distribution", {}), ensure_ascii=False, indent=2, default=str))
        lines.append("  - Распределение train по дикторам:")
        lines.append(json.dumps(split_info.get("train_speaker_distribution", {}), ensure_ascii=False, indent=2, default=str))
        lines.append("  - Распределение test по дикторам:")
        lines.append(json.dumps(split_info.get("test_speaker_distribution", {}), ensure_ascii=False, indent=2, default=str))
        lines.append(f"  - Дикторов в train: {split_info.get('train_speaker_count')}")
        lines.append(f"  - Дикторов в test: {split_info.get('test_speaker_count')}")
        lines.append(f"  - Пересечение дикторов: {split_info.get('speaker_overlap_count')}")

    lines.append("")
    lines.append("Сведения о признаках:")
    lines.append(json.dumps(report.feature_info, ensure_ascii=False, indent=2, default=str))
    lines.append("")
    lines.append("Результаты моделей:")
    for model_name, result in report.model_results.items():
        lines.append("-" * 60)
        lines.append(f"Модель: {model_name}")
        for key in ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "macro_f1", "weighted_f1", "roc_auc_macro", "average_precision_macro", "raw_test_macro_f1", "threshold_cv_macro_f1", "threshold_cv_gain", "cv_mean", "cv_std", "train_macro_f1", "overfit_gap", "avg_inference_time_sec", "complex_score", "test_support"]:
            if key in result:
                lines.append(f"{key}: {result[key]}")
        if "metric_intervals" in result:
            lines.append("95% доверительные интервалы (bootstrap по дикторам):")
            lines.append(json.dumps(result["metric_intervals"], ensure_ascii=False, indent=2, default=str))
        if "best_params" in result:
            lines.append("Параметры:")
            lines.append(json.dumps(result["best_params"], ensure_ascii=False, indent=2, default=str))
        if "classification_report" in result:
            lines.append("Classification report:")
            lines.append(str(result["classification_report"]))
        if "confusion_matrix" in result:
            lines.append("Матрица ошибок:")
            lines.append(str(result["confusion_matrix"]))
        if "per_class_roc_auc" in result:
            lines.append("ROC-AUC по классам:")
            lines.append(json.dumps(result["per_class_roc_auc"], ensure_ascii=False, indent=2))
        if "per_class_average_precision" in result:
            lines.append("PR-AUC / Average Precision по классам:")
            lines.append(json.dumps(result["per_class_average_precision"], ensure_ascii=False, indent=2))
        if "decision_biases" in result:
            lines.append("OOF-смещения решающего правила по классам:")
            lines.append(json.dumps(result["decision_biases"], ensure_ascii=False, indent=2))
        if "warnings" in result and result["warnings"]:
            lines.append("Предупреждения модели:")
            for warning in result["warnings"]:
                lines.append(f"  - {warning}")
    lines.append("=" * 60)
    lines.append(f"Выбранная лучшая модель: {report.best_model_name}")
    lines.append(f"Длительность обучения: {report.duration_sec:.2f} сек.")
    if report.warnings:
        lines.append("Предупреждения:")
        for warning in report.warnings:
            lines.append(f"  - {warning}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
