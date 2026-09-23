from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

from app.features import FeatureExtractor
from app.honest_models import (
    Trainer,
    _select_deployment_model,
    predefined_holdout_indices,
    speaker_disjoint_holdout_indices,
)
from app.models import save_bundle
from app.recognizer import Recognizer


class StaticProbabilityModel:
    def __init__(self, probabilities) -> None:
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, values):
        return np.tile(self.probabilities, (len(values), 1))


class SpeakerDisjointSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        labels = []
        speakers = []
        for speaker_index in range(20):
            for class_index in range(4):
                for _ in range(3):
                    labels.append(class_index)
                    speakers.append(f"speaker_{speaker_index:02d}")
        self.y = np.asarray(labels, dtype=int)
        self.groups = np.asarray(speakers, dtype=object)

    def test_holdout_has_no_speaker_overlap_and_is_reproducible(self) -> None:
        first = speaker_disjoint_holdout_indices(self.y, self.groups, random_state=42)
        second = speaker_disjoint_holdout_indices(self.y, self.groups, random_state=42)
        train_idx, test_idx, info = first
        self.assertTrue(np.array_equal(train_idx, second[0]))
        self.assertTrue(np.array_equal(test_idx, second[1]))
        self.assertEqual(info["speaker_overlap_count"], 0)
        self.assertFalse(set(self.groups[train_idx]) & set(self.groups[test_idx]))
        self.assertEqual(set(np.unique(self.y)), set(np.unique(self.y[test_idx])))

    def test_missing_speaker_id_is_rejected(self) -> None:
        groups = self.groups.copy()
        groups[0] = "unknown"
        with self.assertRaisesRegex(ValueError, "speaker_id"):
            speaker_disjoint_holdout_indices(self.y, groups, random_state=42)

    def test_predefined_holdout_preserves_official_split(self) -> None:
        splits = np.array(["train"] * 12 + ["test"] * 12)
        groups = np.array(
            ["tr1"] * 4 + ["tr2"] * 4 + ["tr3"] * 4
            + ["te1"] * 4 + ["te2"] * 4 + ["te3"] * 4
        )
        y = np.tile(np.array([0, 1, 0, 1]), 6)
        train_idx, test_idx, info = predefined_holdout_indices(y, groups, splits)
        self.assertTrue(np.all(splits[train_idx] == "train"))
        self.assertTrue(np.all(splits[test_idx] == "test"))
        self.assertEqual(info["speaker_overlap_count"], 0)
        self.assertEqual(info["strategy"], "predefined_speaker_disjoint_holdout")


class HonestTrainerTests(unittest.TestCase):
    def test_practical_tie_prefers_simpler_model(self) -> None:
        evaluations = {
            "Logistic Regression": SimpleNamespace(name="Logistic Regression", cv_mean=0.500),
            "Random Forest": SimpleNamespace(name="Random Forest", cv_mean=0.506),
            "MLP": SimpleNamespace(name="MLP", cv_mean=0.509),
        }
        self.assertEqual(_select_deployment_model(evaluations), "Logistic Regression")
        evaluations["MLP"].cv_mean = 0.517
        self.assertEqual(_select_deployment_model(evaluations), "MLP")

    def test_training_reports_grouped_protocol_and_intervals(self) -> None:
        rng = np.random.default_rng(42)
        rows = []
        labels = []
        speakers = []
        for speaker_index in range(20):
            speaker_shift = rng.normal(0.0, 0.15, size=12)
            for class_index, label in enumerate(["anger", "calm", "joy", "sadness"]):
                class_signal = np.zeros(12)
                class_signal[class_index] = 1.5
                for _ in range(3):
                    rows.append(class_signal + speaker_shift + rng.normal(0.0, 0.25, size=12))
                    labels.append(label)
                    speakers.append(f"speaker_{speaker_index:02d}")

        X = np.asarray(rows, dtype=np.float32)
        y = np.asarray(labels)
        trainer = Trainer(
            test_size=0.20,
            random_state=42,
            selected_models=["Logistic Regression"],
        )
        bundle, evaluations = trainer.train_all(
            X,
            y,
            [f"feature_{index}" for index in range(X.shape[1])],
            {"records_count": len(X)},
            {"sample_rate": 16000},
            speaker_ids=speakers,
        )

        self.assertEqual(bundle.best_model_name, "Logistic Regression")
        self.assertFalse(bundle.dataset_info["test_used_for_selection"])
        self.assertEqual(bundle.dataset_info["split_info"]["speaker_overlap_count"], 0)
        evaluation = evaluations["Logistic Regression"]
        self.assertGreaterEqual(evaluation.cv_mean or 0.0, 0.0)
        self.assertIn("macro_f1_95ci", evaluation.metric_intervals)
        self.assertGreater(evaluation.test_support, 0)
        self.assertTrue(bundle.preprocessing_params["preprocessing_in_model"])

        # Regression: sklearn MLP and its threshold wrapper must not be
        # mistaken for the legacy TorchMLP during serialization.
        bundle.models["MLP"] = bundle.models["Logistic Regression"]
        with tempfile.TemporaryDirectory() as tmp:
            model_path = save_bundle(bundle, Path(tmp))
            self.assertTrue(model_path.is_file())


class MultimodalRecognizerTests(unittest.TestCase):
    def test_optional_transcript_fuses_probabilities_and_audio_fallback_works(self) -> None:
        labels = LabelEncoder().fit(["anger", "calm", "joy", "sadness"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "bundle.pkl"
            joblib.dump(
                {
                    "models": {
                        "Emotion2Vec + Logistic Regression": StaticProbabilityModel(
                            [0.10, 0.20, 0.60, 0.10]
                        )
                    },
                    "best_model_name": "Emotion2Vec + Logistic Regression",
                    "label_encoder": labels,
                    "feature_names": ["feature_0", "feature_1"],
                    "scaler": None,
                    "selector": None,
                    "text_model": StaticProbabilityModel([0.80, 0.10, 0.05, 0.05]),
                    "fusion_audio_weight": 0.4,
                    "preprocessing_params": {
                        "preprocessing_in_model": True,
                        "feature_backend": "handcrafted",
                    },
                },
                model_path,
            )
            recognizer = Recognizer(model_path, FeatureExtractor(root / "cache"))
            audio_label, audio_probs, audio_name = recognizer._predict_matrix(
                np.zeros((1, 2), dtype=np.float32)
            )
            fused_label, fused_probs, fused_name = recognizer._predict_matrix(
                np.zeros((1, 2), dtype=np.float32), transcript="тестовая фраза"
            )

            self.assertEqual(audio_label, "joy")
            self.assertAlmostEqual(audio_probs["joy"], 0.60)
            self.assertNotIn("транскрипт", audio_name)
            self.assertEqual(fused_label, "anger")
            self.assertAlmostEqual(fused_probs["anger"], 0.52)
            self.assertIn("транскрипт", fused_name)

    def test_file_uses_auto_transcript_and_falls_back_when_asr_fails(self) -> None:
        from unittest.mock import patch
        from app.asr import ASRSegment, Transcription
        from app.recognizer import RecognitionResult

        labels = LabelEncoder().fit(["anger", "calm", "joy", "sadness"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "speech.wav"
            audio.write_bytes(b"placeholder")
            model_path = root / "bundle.pkl"
            joblib.dump({
                "models": {"Emotion2Vec + Logistic Regression": StaticProbabilityModel([0.1, 0.2, 0.6, 0.1])},
                "best_model_name": "Emotion2Vec + Logistic Regression",
                "label_encoder": labels,
                "feature_names": ["feature_0", "feature_1"],
                "scaler": None, "selector": None,
                "text_model": StaticProbabilityModel([0.8, 0.1, 0.05, 0.05]),
                "fusion_audio_weight": 0.4,
                "preprocessing_params": {"preprocessing_in_model": True, "feature_backend": "handcrafted"},
            }, model_path)
            recognizer = Recognizer(model_path, FeatureExtractor(root / "cache"))
            transcript = Transcription("привет", (ASRSegment(0.0, 1.0, "привет"),))
            with patch("app.recognizer.load_audio", return_value=(np.zeros(16000, dtype=np.float32), 16000)), \
                 patch("app.recognizer.preprocess_signal", side_effect=lambda y, *args, **kwargs: y), \
                 patch("app.recognizer.analyze_quality", return_value=type("Quality", (), {"warnings": []})()), \
                 patch("app.recognizer.split_long_signal", return_value=[np.zeros(16000, dtype=np.float32)]), \
                 patch.object(recognizer.extractor, "extract_signal", return_value=(np.zeros(2, dtype=np.float32), [])), \
                 patch.object(recognizer, "_transcribe_file", return_value=transcript):
                result = recognizer.recognize_file(audio)
            self.assertIsInstance(result, RecognitionResult)
            self.assertEqual(result.predicted_emotion, "anger")
            self.assertEqual(result.transcript, "привет")
            self.assertEqual(result.transcript_source, "автоматический")

            with patch("app.recognizer.load_audio", return_value=(np.zeros(16000, dtype=np.float32), 16000)), \
                 patch("app.recognizer.preprocess_signal", side_effect=lambda y, *args, **kwargs: y), \
                 patch("app.recognizer.analyze_quality", return_value=type("Quality", (), {"warnings": []})()), \
                 patch("app.recognizer.split_long_signal", return_value=[np.zeros(16000, dtype=np.float32)]), \
                 patch.object(recognizer.extractor, "extract_signal", return_value=(np.zeros(2, dtype=np.float32), [])), \
                 patch.object(recognizer, "_transcribe_file", side_effect=RuntimeError("нет ASR")):
                fallback = recognizer.recognize_file(audio)
            self.assertEqual(fallback.predicted_emotion, "joy")
            self.assertTrue(any("только аудиомодель" in item for item in fallback.warnings))


if __name__ == "__main__":
    unittest.main()
