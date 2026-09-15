"""Общий интерфейс вероятностных классификаторов проекта."""

from typing import Protocol

import numpy as np


class SupportsProbabilityPrediction(Protocol):

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        ...


class ProbabilityPredictionMixin:

    def predict(self: SupportsProbabilityPrediction, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(features), axis=1)
