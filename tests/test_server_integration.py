"""Проверки метаданных при публикации модели."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import joblib

from app.server_integration import ServerIntegrationError, model_publication_metadata, validate_admin_url


class ServerIntegrationTests(unittest.TestCase):
    def test_admin_key_requires_https_outside_localhost(self):
        self.assertEqual(validate_admin_url("http://127.0.0.1:8000"), "http://127.0.0.1:8000")
        self.assertEqual(validate_admin_url("https://example.org"), "https://example.org")
        with self.assertRaises(ServerIntegrationError):
            validate_admin_url("http://example.org")
        with self.assertRaises(ServerIntegrationError):
            validate_admin_url("https://user:pass" + chr(64) + "example.org")

    def test_publication_metadata_does_not_claim_unmeasured_class_calibration(self):
        with tempfile.TemporaryDirectory() as folder:
            model_path = Path(folder) / "model.pkl"
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

        self.assertEqual(model_name, "Random Forest")
        self.assertEqual(config["feature_count"], 2)
        self.assertEqual(thresholds["per_emotion"], {})
        self.assertEqual(thresholds["calibration"], "fixed_heuristic")


if __name__ == "__main__":
    unittest.main()
