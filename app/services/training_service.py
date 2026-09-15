"""Полный сценарий обучения моделей без зависимости от PySide6."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from ..config import AppConfig, ProjectPaths
from ..dataset import DatasetLoadResult
from ..features import FeatureExtractor, FeatureMatrix
from ..ml.persistence import save_bundle
from ..ml.training import Trainer
from ..ml.types import ModelEvaluation, TrainingBundle
from ..storage import ExperimentDB, TrainingReport, write_training_report
from .experiment_signature import build_experiment_signature

ProgressCallback = Callable[[int, str], None]


class TrainingService:

    def __init__(
        self,
        dataset_result: DatasetLoadResult,
        config: AppConfig,
        project_paths: ProjectPaths,
        selected_models: Iterable[str] | None = None,
        parameter_mode: str | None = None,
        manual_parameters: dict[str, Any] | None = None,
    ) -> None:
        self.dataset_result = dataset_result
        self.config = config
        self.project_paths = project_paths
        self.selected_models = list(selected_models or config.training_models)
        self.parameter_mode = parameter_mode or config.parameter_mode
        self.manual_parameters = manual_parameters or {}

    def execute(self, progress_callback: ProgressCallback | None = None) -> dict[str, Any]:
        started_at = time.perf_counter()
        progress = progress_callback or (lambda _value, _message: None)

        progress(3, "Проверка истории экспериментов")
        experiment_database = ExperimentDB(
            self.project_paths.database / "experiments.sqlite"
        )
        signature = build_experiment_signature(
            self.dataset_result,
            self.config,
            self.selected_models,
            self.parameter_mode,
            self.manual_parameters,
        )
        reusable_result = self._find_reusable_result(experiment_database, signature)
        if reusable_result is not None:
            progress(100, "Такой эксперимент уже выполнялся ранее; повторное обучение пропущено")
            return reusable_result

        feature_matrix = self._extract_features(progress)
        trainer = self._create_trainer()
        progress(45, "Обучение выбранных моделей и расчёт метрик")
        training_bundle, evaluations = trainer.train_all(
            feature_matrix.feature_matrix,
            feature_matrix.emotion_labels,
            feature_matrix.feature_names,
            self._dataset_metadata(feature_matrix),
            self._preprocessing_metadata(),
            speaker_ids=feature_matrix.speaker_ids,
        )

        progress(85, "Сохранение модели и отчёта")
        model_path = save_bundle(training_bundle, self.project_paths.models)
        duration_seconds = time.perf_counter() - started_at
        report = self._build_report(
            feature_matrix,
            training_bundle,
            evaluations,
            duration_seconds,
        )
        report_path = write_training_report(report, self.project_paths.reports)
        experiment_database.add_experiment(
            dataset_path=self.dataset_result.original_path,
            selected_models=self.selected_models,
            parameters={name: evaluation.best_params for name, evaluation in evaluations.items()},
            metrics={name: evaluation.to_dict() for name, evaluation in evaluations.items()},
            best_model_name=training_bundle.best_model_name,
            model_path=str(model_path),
            report_path=str(report_path),
            duration_sec=time.perf_counter() - started_at,
            warnings=report.warnings,
            experiment_signature=signature,
        )
        progress(100, "Обучение завершено")
        return {
            "bundle": training_bundle,
            "evaluations": evaluations,
            "model_path": str(model_path),
            "report_path": str(report_path),
            "matrix_errors": feature_matrix.errors,
        }

    def _find_reusable_result(
        self,
        experiment_database: ExperimentDB,
        signature: str,
    ) -> dict[str, Any] | None:
        history_row = experiment_database.find_by_signature(signature)
        if history_row is None:
            return None
        model_path = str(history_row.get("model_path", ""))
        if not model_path or not Path(model_path).exists():
            return None
        best_model_name = str(history_row.get("best_model_name") or "").strip()
        if not best_model_name:
            return None
        try:
            metrics = json.loads(history_row.get("metrics_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            metrics = {}
        return {
            "skipped": True,
            "reason": (
                "Эксперимент с таким датасетом, моделями, параметрами и настройками "
                "уже есть в истории."
            ),
            "model_path": model_path,
            "report_path": history_row.get("report_path", ""),
            "metrics": metrics,
            "best_model_name": best_model_name,
            "history_row": history_row,
        }

    def _extract_features(self, progress: ProgressCallback) -> FeatureMatrix:
        progress(5, "Подготовка извлечения признаков")
        extractor = FeatureExtractor(
            self.project_paths.cache,
            sample_rate=self.config.sample_rate,
            denoise=self.config.denoise,
            trim=self.config.trim_silence,
            normalize=self.config.normalize_amplitude,
        )
        progress(20, "Извлечение акустических признаков")
        feature_matrix = extractor.build_matrix(self.dataset_result.dataframe)
        error_ratio = len(feature_matrix.errors) / max(len(self.dataset_result.dataframe), 1)
        if error_ratio > 0.05:
            raise RuntimeError(
                "Ошибок извлечения признаков больше 5%. Обучение остановлено.\n"
                + "\n".join(feature_matrix.errors[:20])
            )
        return feature_matrix

    def _create_trainer(self) -> Trainer:
        return Trainer(
            test_size=self.config.test_size,
            use_gpu=self.config.use_gpu_if_available,
            feature_selection=self.config.feature_selection,
            random_state=self.config.random_state,
            hyperparameter_mode=self.config.hyperparameter_mode,
            selected_models=self.selected_models,
            parameter_mode=self.parameter_mode,
            manual_params=self.manual_parameters,
        )

    def _dataset_metadata(self, feature_matrix: FeatureMatrix) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_result.original_path,
            "records_count": feature_matrix.feature_matrix.shape[0],
            "class_distribution": self.dataset_result.distribution,
            "excluded_rows": self.dataset_result.excluded_rows,
        }

    def _preprocessing_metadata(self) -> dict[str, Any]:
        return {
            "sample_rate": self.config.sample_rate,
            "normalize_amplitude": self.config.normalize_amplitude,
            "denoise": self.config.denoise,
            "trim_silence": self.config.trim_silence,
            "test_size": self.config.test_size,
        }

    def _build_report(
        self,
        feature_matrix: FeatureMatrix,
        training_bundle: TrainingBundle,
        evaluations: dict[str, ModelEvaluation],
        duration_seconds: float,
    ) -> TrainingReport:
        return TrainingReport(
            dataset_path=self.dataset_result.original_path,
            processed_files=feature_matrix.feature_matrix.shape[0],
            class_distribution=self.dataset_result.distribution,
            feature_info={
                "feature_count": feature_matrix.feature_matrix.shape[1],
                "feature_names_sample": feature_matrix.feature_names[:25],
                "cache_dir": str(self.project_paths.cache),
                "split_info": training_bundle.dataset_info.get("split_info", {}),
                "validation_split_info": training_bundle.dataset_info.get(
                    "validation_split_info", {}
                ),
                "model_selection_metric": training_bundle.dataset_info.get(
                    "model_selection_metric"
                ),
                "model_selection_scores": training_bundle.dataset_info.get(
                    "model_selection_scores", {}
                ),
            },
            model_results={
                name: evaluation.to_dict() for name, evaluation in evaluations.items()
            },
            best_model_name=training_bundle.best_model_name,
            warnings=(
                self.dataset_result.warnings
                + feature_matrix.errors
                + training_bundle.dataset_info.get("split_warnings", [])
            ),
            duration_sec=duration_seconds,
        )
