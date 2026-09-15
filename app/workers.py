"""Фоновые задачи Qt для загрузки данных, обучения и распознавания."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from PySide6.QtCore import QObject, Signal, Slot

from .config import AppConfig, ProjectPaths
from .dataset import DatasetLoader, DatasetLoadResult
from .services.training_service import TrainingService


class FunctionWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, operation: Callable[..., Any], *args: Any, **kwargs: Any):
        super().__init__()
        self.operation = operation
        self.operation_args = args
        self.operation_kwargs = kwargs

    @Slot()
    def run(self) -> None:
        try:
            result = self.operation(*self.operation_args, **self.operation_kwargs)
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class DatasetLoadWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        csv_path: str,
        min_duration: float = 3.0,
        emotion_mapping: dict[str, str] | None = None,
        validate_audio_quality: bool = True,
        sample_rate: int = 16_000,
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.min_duration = min_duration
        self.emotion_mapping = emotion_mapping
        self.validate_audio_quality = validate_audio_quality
        self.sample_rate = sample_rate

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit(10, "Чтение CSV-файла")
            loader = DatasetLoader(
                self.min_duration,
                validate_audio_quality=self.validate_audio_quality,
                sample_rate=self.sample_rate,
            )
            result = loader.load_csv(
                self.csv_path,
                emotion_mapping=self.emotion_mapping,
                progress_callback=self.progress.emit,
            )
            self.progress.emit(100, "Датасет загружен")
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))

class TrainingWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        dataset_result: DatasetLoadResult,
        config: AppConfig,
        project_paths: ProjectPaths,
        selected_models: Iterable[str] | None = None,
        parameter_mode: str | None = None,
        manual_params: dict | None = None,
    ) -> None:
        super().__init__()
        self.dataset_result = dataset_result
        self.config = config
        self.project_paths = project_paths
        self.selected_models = list(selected_models or config.training_models)
        self.parameter_mode = parameter_mode or config.parameter_mode
        self.manual_params = manual_params or {}

    @Slot()
    def run(self) -> None:
        try:
            training_service = TrainingService(
                dataset_result=self.dataset_result,
                config=self.config,
                project_paths=self.project_paths,
                selected_models=self.selected_models,
                parameter_mode=self.parameter_mode,
                manual_parameters=self.manual_params,
            )
            result = training_service.execute(self.progress.emit)
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))
