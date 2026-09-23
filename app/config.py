from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class AppConfig:
    theme: str = "system"
    sample_rate: int = 16000
    mono: bool = True
    min_train_duration_sec: float = 3.0
    max_segment_duration_sec: float = 10.0
    record_duration_sec: int = 5
    test_size: float = 0.20
    parameter_mode: str = "auto"
    training_models: List[str] = field(
        default_factory=lambda: ["Logistic Regression", "Random Forest", "MLP"]
    )
    denoise: bool = False
    trim_silence: bool = True
    normalize_amplitude: bool = True
    feature_selection: str = "auto"
    validate_external_audio: bool = True
    random_state: int | None = 42
    input_device_index: int | None = None


    manual_svm_c: float = 1.0
    manual_svm_gamma: str = "scale"
    manual_svm_kernel: str = "rbf"
    manual_logistic_c: float = 1.0
    base_dir: str = "."
    models_dir: str = "models"
    records_dir: str = "records"
    datasets_dir: str = "datasets"
    cache_dir: str = "cache"
    reports_dir: str = "reports"
    logs_dir: str = "logs"
    exports_dir: str = "exports"
    database_dir: str = "database"

    @classmethod
    def load(cls, path: str | Path = "config.json") -> "AppConfig":
        path = Path(path)
        if not path.exists():
            cfg = cls()
            cfg.save(path)
            return cfg
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            allowed = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(**allowed)
        except Exception:
            cfg = cls()
            backup = path.with_suffix(".broken.json")
            try:
                path.replace(backup)
            except Exception:
                pass
            cfg.save(path)
            return cfg

    def save(self, path: str | Path = "config.json") -> None:
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProjectPaths:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.base = Path(cfg.base_dir).resolve()
        self.models = self.base / cfg.models_dir
        self.records = self.base / cfg.records_dir
        self.datasets = self.base / cfg.datasets_dir
        self.cache = self.base / cfg.cache_dir
        self.reports = self.base / cfg.reports_dir
        self.logs = self.base / cfg.logs_dir
        self.exports = self.base / cfg.exports_dir
        self.database = self.base / cfg.database_dir

    def ensure(self) -> None:
        for path in [self.models, self.records, self.datasets, self.cache, self.reports, self.logs, self.exports, self.database]:
            path.mkdir(parents=True, exist_ok=True)
        (self.datasets / "user_dataset").mkdir(parents=True, exist_ok=True)
        for path in [self.models, self.records, self.datasets, self.cache, self.reports, self.logs, self.exports, self.database]:
            keep = path / ".gitkeep"
            if not keep.exists():
                keep.write_text("", encoding="utf-8")
