"""Ансамбли, объединяющие вероятности нескольких базовых моделей."""

from typing import Any

import numpy as np

from .probability import ProbabilityPredictionMixin


class ProbabilityEnsemble(ProbabilityPredictionMixin):

    def __init__(self, models: dict[str, Any], weights: dict[int, dict[str, float]]):
        if len(models) < 2:
            raise ValueError("Для объединённой модели нужно минимум две базовые модели.")
        self.models = models
        self.weights = weights

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities_by_model = {name: model.predict_proba(features) for name, model in self.models.items()}
        first = next(iter(probabilities_by_model.values()))
        combined_probabilities = np.zeros_like(first, dtype=np.float64)
        model_names = list(probabilities_by_model.keys())
        uniform = 1.0 / max(len(model_names), 1)
        for class_idx in range(first.shape[1]):
            class_weights = self.weights.get(class_idx, {})
            for name in model_names:
                combined_probabilities[:, class_idx] += float(class_weights.get(name, uniform)) * probabilities_by_model[name][:, class_idx]
        row_totals = combined_probabilities.sum(axis=1, keepdims=True)
        row_totals[row_totals == 0] = 1.0
        return combined_probabilities / row_totals
