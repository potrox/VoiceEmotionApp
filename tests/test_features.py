"""Проверки фиксированной схемы акустических признаков."""

from __future__ import annotations

import numpy as np

from app.features import FeatureExtractor


def test_failed_component_keeps_its_declared_feature_width() -> None:
    def failed_extractor() -> np.ndarray:
        raise RuntimeError("synthetic extraction failure")

    values, names = FeatureExtractor._extract_component(
        "spectral_contrast", 7, failed_extractor
    )

    assert len(values) == 28
    assert len(names) == 28
    assert np.allclose(values, 0.0)
