"""Поддерживаемые модели, ансамбли и пространства настройки параметров."""

from typing import Any

BASE_MODEL_NAMES = frozenset({"SVM", "MLP", "Random Forest"})

ENSEMBLE_DEFINITIONS: dict[str, tuple[str, ...]] = {
    "Ансамбль SVM + MLP": ("SVM", "MLP"),
    "Ансамбль SVM + RandomForest": ("SVM", "Random Forest"),
    "Ансамбль MLP + RandomForest": ("MLP", "Random Forest"),
    "Ансамбль SVM + MLP + RandomForest": (
        "SVM",
        "MLP",
        "Random Forest",
    ),
}

SVM_PARAMETER_GRIDS: dict[str, dict[str, list[Any]]] = {
    "quality": {
        "classifier__C": [0.5, 1.0, 5.0, 10.0],
        "classifier__kernel": ["rbf", "linear"],
        "classifier__gamma": ["scale"],
    },
    "fast": {
        "classifier__C": [1.0, 5.0, 10.0],
        "classifier__kernel": ["rbf", "linear"],
        "classifier__gamma": ["scale"],
    },
}

RANDOM_FOREST_PARAMETER_GRIDS: dict[str, dict[str, list[Any]]] = {
    "quality": {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [None, 12, 24],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
    },
    "fast": {
        "classifier__n_estimators": [150, 300],
        "classifier__max_depth": [None, 16],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1],
    },
}

MLP_CANDIDATES: dict[str, tuple[dict[str, Any], ...]] = {
    "quality": (
        {"hidden_sizes": (64,), "learning_rate": 1e-3, "epochs": 80, "batch_size": 32},
        {"hidden_sizes": (128,), "learning_rate": 1e-3, "epochs": 100, "batch_size": 32},
        {"hidden_sizes": (128, 64), "learning_rate": 1e-3, "epochs": 100, "batch_size": 32},
        {"hidden_sizes": (128, 64), "learning_rate": 5e-4, "epochs": 140, "batch_size": 32},
        {"hidden_sizes": (256, 128), "learning_rate": 8e-4, "epochs": 120, "batch_size": 32},
        {"hidden_sizes": (256, 128), "learning_rate": 5e-4, "epochs": 150, "batch_size": 64},
        {"hidden_sizes": (256, 128, 64), "learning_rate": 5e-4, "epochs": 140, "batch_size": 32},
        {"hidden_sizes": (256, 128, 64), "learning_rate": 3e-4, "epochs": 180, "batch_size": 64},
    ),
    "fast": (
        {"hidden_sizes": (64,), "learning_rate": 1e-3, "epochs": 70, "batch_size": 32},
        {"hidden_sizes": (128, 64), "learning_rate": 1e-3, "epochs": 90, "batch_size": 32},
        {"hidden_sizes": (256, 128), "learning_rate": 8e-4, "epochs": 100, "batch_size": 32},
    ),
}
