import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from scripts.evaluation_components import _metrics as probability_metrics
from scripts.experiment_common import CLASSES, classification_metrics, sample_train
from scripts.multimodal_components import _audio_model, _cache_path, _text_model


class ExperimentComponentsTests(unittest.TestCase):
    def test_sampling_is_balanced_train_only_and_repeatable(self):
        rows = [
            {"file_path": f"{emotion}-{index}", "emotion": emotion, "dataset_split": "train"}
            for emotion in CLASSES
            for index in range(3)
        ]
        rows += [
            {"file_path": f"heldout-{emotion}", "emotion": emotion, "dataset_split": "test"}
            for emotion in CLASSES
        ]
        data = pd.DataFrame(rows)
        first = sample_train(data, max_per_class=2, seed=42)
        second = sample_train(data, max_per_class=2, seed=42)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.emotion.value_counts().to_dict(), {name: 2 for name in CLASSES})
        self.assertTrue((first.dataset_split == "train").all())

    def test_classification_metrics_and_selective_coverage(self):
        true = np.array([0, 1, 2, 3])
        accepted = np.array([True, False, True, False])
        report = classification_metrics(true, true.copy(), accepted)
        self.assertEqual(report["macro_f1"], 1.0)
        self.assertEqual(report["coverage"], 0.5)
        self.assertEqual(report["accepted_accuracy"], 1.0)
        self.assertEqual(set(report["per_class"]), set(CLASSES))

    def test_model_factories_keep_frozen_parameters(self):
        audio = _audio_model(0.1, 768, 42)
        text = _text_model(0.1, 42)
        self.assertEqual(audio.named_steps["selector"].k, 256)
        self.assertEqual(audio.named_steps["model"].C, 0.1)
        self.assertEqual(text.named_steps["tfidf"].ngram_range, (3, 5))
        self.assertEqual(text.named_steps["model"].C, 0.1)

    def test_embedding_cache_path_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "short.wav"
            file.write_bytes(b"test-audio")
            cache = Path(directory) / "cache"
            self.assertEqual(_cache_path(cache, str(file)), _cache_path(cache, str(file)))

    def test_probability_report_keeps_multiclass_metrics(self):
        encoder = LabelEncoder().fit(CLASSES)
        true = np.arange(len(CLASSES))
        probabilities = np.eye(len(CLASSES))
        report = probability_metrics(true, probabilities, encoder)
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["macro_f1"], 1.0)
        self.assertEqual(report["confusion_matrix"], np.eye(4, dtype=int).tolist())


if __name__ == "__main__":
    unittest.main()
