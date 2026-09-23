from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, Signal, Slot

from .dataset import DatasetLoader
from .features import FeatureExtractor
from .honest_models import Trainer
from .models import save_bundle
from .storage import ExperimentDB, TrainingReport, write_training_report


class FunctionWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, func: Callable[..., Any], *args: Any, **kwargs: Any):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    @Slot()
    def run(self) -> None:
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class DatasetLoadWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, csv_path: str, min_duration: float = 3.0, emotion_mapping: dict | None = None, validate_audio_quality: bool = True):
        super().__init__()
        self.csv_path = csv_path
        self.min_duration = min_duration
        self.emotion_mapping = emotion_mapping
        self.validate_audio_quality = validate_audio_quality

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit(10, "Чтение CSV-файла")
            loader = DatasetLoader(self.min_duration, validate_audio_quality=self.validate_audio_quality)
            result = loader.load_csv(self.csv_path, emotion_mapping=self.emotion_mapping, progress_callback=self.progress.emit)
            self.progress.emit(100, "Датасет загружен")
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))




def _file_fingerprint(path_value: str) -> dict:
    path = Path(path_value)
    try:
        stat = path.stat()
        return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except Exception:
        return {"path": str(path_value), "size": None, "mtime_ns": None}


def build_experiment_signature(dataset_result, cfg, selected_models, parameter_mode: str, manual_params: dict) -> str:
    df = dataset_result.dataframe.copy()
    file_rows = []
    if "file_path" in df.columns:
        for _, row in df.sort_values(by=["file_path", "emotion"]).iterrows():
            file_rows.append({
                "file": _file_fingerprint(str(row.get("file_path", ""))),
                "emotion": str(row.get("emotion", "")),
                "speaker_id": str(row.get("speaker_id", "")),
                "dataset_split": str(row.get("dataset_split", "")),
            })
    payload = {
        "dataset_path": str(dataset_result.original_path),
        "records_count": int(len(df)),
        "files": file_rows,
        "selected_models": sorted(list(selected_models)),
        "parameter_mode": parameter_mode,
        "manual_params": manual_params if parameter_mode == "manual" else {},
        "training_config": {
            "test_size": cfg.test_size,
            "feature_selection": cfg.feature_selection,
            "sample_rate": cfg.sample_rate,
            "denoise": cfg.denoise,
            "trim_silence": cfg.trim_silence,
            "normalize_amplitude": cfg.normalize_amplitude,
            "random_state": cfg.random_state,
            "evaluation_protocol": "speaker_independent_grouped_cv_v4_oof_thresholds",
            "feature_profile": "compact_v1_k96_median",
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TrainingWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, dataset_result, cfg, paths, selected_models=None, parameter_mode: str | None = None, manual_params: dict | None = None):
        super().__init__()
        self.dataset_result = dataset_result
        self.cfg = cfg
        self.paths = paths
        self.selected_models = selected_models or cfg.training_models
        self.parameter_mode = parameter_mode or cfg.parameter_mode
        self.manual_params = manual_params or {}

    @Slot()
    def run(self) -> None:
        start = time.perf_counter()
        try:
            self.progress.emit(3, "Проверка истории экспериментов")
            db = ExperimentDB(self.paths.database / "experiments.sqlite")
            signature = build_experiment_signature(self.dataset_result, self.cfg, self.selected_models, self.parameter_mode, self.manual_params)
            previous = db.find_by_signature(signature)
            previous_model_path = str(previous.get("model_path", "")) if previous else ""
            if previous and previous_model_path and Path(previous_model_path).exists():
                self.progress.emit(100, "Такой эксперимент уже выполнялся ранее; повторное обучение пропущено")
                self.finished.emit({
                    "skipped": True,
                    "reason": "Эксперимент с таким датасетом, моделями, параметрами и настройками уже есть в истории.",
                    "model_path": previous.get("model_path", ""),
                    "report_path": previous.get("report_path", ""),
                    "metrics": json.loads(previous.get("metrics_json") or "{}"),
                    "history_row": previous,
                })
                return
            self.progress.emit(5, "Подготовка извлечения признаков")
            extractor = FeatureExtractor(
                self.paths.cache,
                sample_rate=self.cfg.sample_rate,
                denoise=self.cfg.denoise,
                trim=self.cfg.trim_silence,
                normalize=self.cfg.normalize_amplitude,
            )
            self.progress.emit(20, "Извлечение акустических признаков")
            matrix = extractor.build_matrix(
                self.dataset_result.dataframe,
                progress_callback=lambda done, total: self.progress.emit(
                    20 + int(25 * done / max(total, 1)),
                    f"Извлечение признаков: {done} из {total}",
                ),
            )
            if matrix.errors:
                error_ratio = len(matrix.errors) / max(len(self.dataset_result.dataframe), 1)
                if error_ratio > 0.05:
                    raise RuntimeError("Ошибок извлечения признаков больше 5%. Обучение остановлено.\n" + "\n".join(matrix.errors[:20]))
            self.progress.emit(45, "Обучение выбранных моделей и расчёт метрик")
            trainer = Trainer(
                test_size=self.cfg.test_size,
                feature_selection=self.cfg.feature_selection,
                random_state=self.cfg.random_state,
                selected_models=self.selected_models,
                parameter_mode=self.parameter_mode,
                manual_params=self.manual_params,
            )
            dataset_info = {
                "dataset_path": self.dataset_result.original_path,
                "records_count": matrix.X.shape[0],
                "class_distribution": self.dataset_result.distribution,
                "excluded_rows": self.dataset_result.excluded_rows,
            }
            preprocessing_params = {
                "sample_rate": self.cfg.sample_rate,
                "normalize_amplitude": self.cfg.normalize_amplitude,
                "denoise": self.cfg.denoise,
                "trim_silence": self.cfg.trim_silence,
                "test_size": self.cfg.test_size,
                "evaluation_protocol": "speaker_independent_grouped_cv",
            }
            bundle, evaluations = trainer.train_all(
                matrix.X,
                matrix.y,
                matrix.feature_names,
                dataset_info,
                preprocessing_params,
                speaker_ids=matrix.speaker_ids,
                split_labels=matrix.split_labels,
            )
            self.progress.emit(85, "Сохранение модели и отчёта")
            model_path = save_bundle(bundle, self.paths.models)
            report = TrainingReport(
                dataset_path=self.dataset_result.original_path,
                processed_files=matrix.X.shape[0],
                class_distribution=self.dataset_result.distribution,
                feature_info={
                    "feature_count": matrix.X.shape[1],
                    "feature_names_sample": matrix.feature_names[:25],
                    "cache_dir": str(self.paths.cache),
                    "split_info": bundle.dataset_info.get("split_info", {}),
                    "validation_split_info": bundle.dataset_info.get("validation_split_info", {}),
                },
                model_results={k: v.to_dict() for k, v in evaluations.items()},
                best_model_name=bundle.best_model_name,
                warnings=self.dataset_result.warnings + matrix.errors + bundle.dataset_info.get("split_warnings", []),
                duration_sec=time.perf_counter() - start,
            )
            report_path = write_training_report(report, self.paths.reports)
            db.add_experiment(
                dataset_path=self.dataset_result.original_path,
                selected_models=self.selected_models,
                parameters={name: ev.best_params for name, ev in evaluations.items()},
                metrics={name: ev.to_dict() for name, ev in evaluations.items()},
                model_path=str(model_path),
                report_path=str(report_path),
                duration_sec=time.perf_counter() - start,
                warnings=report.warnings,
                experiment_signature=signature,
            )
            self.progress.emit(100, "Обучение завершено")
            self.finished.emit({"bundle": bundle, "evaluations": evaluations, "model_path": str(model_path), "report_path": str(report_path), "matrix_errors": matrix.errors})
        except Exception as exc:
            self.failed.emit(str(exc))
