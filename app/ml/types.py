"""Структуры данных, которыми обмениваются этапы обучения."""

from dataclasses import dataclass, field
from typing import Any

from sklearn.preprocessing import LabelEncoder, StandardScaler

@dataclass
class ModelEvaluation:

    name: str
    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    macro_f1: float
    weighted_f1: float
    classification_report: str
    confusion_matrix: list[list[int]]
    per_class_f1: dict[str, float]
    train_macro_f1: float
    overfit_gap: float
    cv_mean: float | None
    cv_std: float | None
    avg_inference_time_sec: float
    best_params: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class TrainingBundle:

    models: dict[str, Any]
    best_model_name: str
    scaler: StandardScaler
    label_encoder: LabelEncoder
    feature_names: list[str]
    selector: Any
    preprocessing_params: dict[str, Any]
    metrics: dict[str, Any]
    dataset_info: dict[str, Any]
    ensemble_weights: dict[str, Any]
    created_at: str
