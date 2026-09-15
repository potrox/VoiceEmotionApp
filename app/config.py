"""Настройки приложения и вычисление путей рабочих каталогов."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AppConfig:

    theme: str = "system"
    sample_rate: int = 16_000
    mono: bool = True
    min_train_duration_sec: float = 3.0
    max_segment_duration_sec: float = 10.0
    record_duration_sec: int = 5
    test_size: float = 0.30
    hyperparameter_mode: str = "quality"
    parameter_mode: str = "auto"
    training_models: list[str] = field(
        default_factory=lambda: [
            "SVM",
            "MLP",
            "Random Forest",
            "Ансамбль SVM + MLP",
            "Ансамбль SVM + RandomForest",
            "Ансамбль MLP + RandomForest",
            "Ансамбль SVM + MLP + RandomForest",
        ]
    )
    use_gpu_if_available: bool = True
    denoise: bool = True
    trim_silence: bool = True
    normalize_amplitude: bool = True
    feature_selection: str = "auto"
    validate_external_audio: bool = True
    random_state: int = 42
    input_device_index: int | None = None

    manual_svm_c: float = 1.0
    manual_svm_gamma: str = "scale"
    manual_svm_kernel: str = "rbf"
    manual_rf_n_estimators: int = 200
    manual_rf_max_depth: int = 0
    manual_rf_min_samples_split: int = 2
    manual_rf_min_samples_leaf: int = 1
    manual_mlp_hidden_layers: str = "128,64"
    manual_mlp_learning_rate: float = 0.001
    manual_mlp_epochs: int = 80
    manual_mlp_batch_size: int = 32

    base_dir: str = "."
    models_dir: str = "models"
    records_dir: str = "records"
    datasets_dir: str = "datasets"
    cache_dir: str = "cache"
    reports_dir: str = "reports"
    logs_dir: str = "logs"
    exports_dir: str = "exports"
    database_dir: str = "database"
    server_url: str = "http://127.0.0.1:8000"
    server_dir: str = "server"

    @classmethod
    def load(cls, path: str | Path = "config.json") -> "AppConfig":
        config_path = Path(path)
        if not config_path.exists():
            config = cls()
            config.save(config_path)
            return config
        try:
            raw_settings = json.loads(config_path.read_text(encoding="utf-8"))
            known_fields = cls.__dataclass_fields__
            settings = {
                name: value
                for name, value in raw_settings.items()
                if name in known_fields
            }
            return cls(**settings)
        except (AttributeError, OSError, TypeError, ValueError, json.JSONDecodeError):
            config = cls()
            broken_path = config_path.with_suffix(".broken.json")
            try:
                config_path.replace(broken_path)
            except OSError:
                pass
            config.save(config_path)
            return config

    def save(self, path: str | Path = "config.json") -> None:
        Path(path).write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProjectPaths:

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.base = Path(config.base_dir).resolve()
        self.models = self.base / config.models_dir
        self.records = self.base / config.records_dir
        self.datasets = self.base / config.datasets_dir
        self.cache = self.base / config.cache_dir
        self.reports = self.base / config.reports_dir
        self.logs = self.base / config.logs_dir
        self.exports = self.base / config.exports_dir
        self.database = self.base / config.database_dir

    @property
    def working_directories(self) -> tuple[Path, ...]:
        return (
            self.models,
            self.records,
            self.datasets,
            self.cache,
            self.reports,
            self.logs,
            self.exports,
            self.database,
        )

    def ensure(self) -> None:
        for directory in self.working_directories:
            directory.mkdir(parents=True, exist_ok=True)
            keep_file = directory / ".gitkeep"
            if not keep_file.exists():
                keep_file.write_text("", encoding="utf-8")
        (self.datasets / "user_dataset").mkdir(parents=True, exist_ok=True)
