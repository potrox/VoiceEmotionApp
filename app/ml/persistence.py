"""Сохранение, загрузка и подготовка обученных моделей."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..constants import FEATURE_SCHEMA_VERSION
from ..storage import timestamp
from .types import TrainingBundle


def save_bundle(bundle: TrainingBundle, models_dir: str | Path) -> Path:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"model_{timestamp()}.pkl"
    serializable = bundle.__dict__.copy()

    joblib.dump(serializable, path)
    return path


def load_bundle(path: str | Path) -> dict[str, Any]:
    try:
        bundle = joblib.load(path)
    except (ModuleNotFoundError, AttributeError) as exc:
        raise ValueError(
            "Пакет модели создан несовместимой версией приложения. "
            "Загрузите датасет и обучите модель заново."
        ) from exc
    saved_schema_version = bundle.get("preprocessing_params", {}).get(
        "feature_schema_version"
    )
    if saved_schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            "Модель создана с устаревшей схемой признаков. "
            "Загрузите датасет и обучите модель заново."
        )
    return bundle


def transform_features(bundle: dict[str, Any], feature_matrix: np.ndarray) -> np.ndarray:
    expected_feature_count = len(bundle.get("feature_names", []))
    if feature_matrix.ndim != 2 or feature_matrix.shape[1] != expected_feature_count:
        raise ValueError(
            "Матрица признаков не соответствует сохранённой модели: "
            f"ожидалась форма (*, {expected_feature_count}), получена "
            f"{feature_matrix.shape}."
        )
    transformed_features = bundle["scaler"].transform(feature_matrix)
    selector = bundle.get("selector")
    if selector is not None:
        transformed_features = selector.transform(transformed_features)
    return transformed_features
