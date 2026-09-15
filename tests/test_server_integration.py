"""Проверки метаданных при публикации модели."""

from __future__ import annotations

from types import SimpleNamespace

import joblib

from app.server_integration import model_publication_metadata


def test_publication_metadata_does_not_claim_unmeasured_class_calibration(
    tmp_path,
) -> None:
    model_path = tmp_path / "model.pkl"
    joblib.dump(
        {
            "best_model_name": "RandomForest",
            "label_encoder": SimpleNamespace(classes_=["calm", "joy"]),
            "feature_names": ["feature_0", "feature_1"],
            "preprocessing_params": {"sample_rate": 16_000},
            "metrics": {
                "RandomForest": {
                    "per_class_f1": {"calm": 0.1, "joy": 0.9},
                }
            },
        },
        model_path,
    )

    model_name, config, thresholds = model_publication_metadata(model_path)

    assert model_name == "Random Forest"
    assert config["feature_count"] == 2
    assert thresholds["per_emotion"] == {}
    assert thresholds["calibration"] == "fixed_heuristic"
