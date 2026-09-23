from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.storage import timestamp


MODEL_NAME = "Emotion2Vec + Logistic Regression"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deployable optional-transcript bundle")
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    components = joblib.load(args.components)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    report = metrics["previous_official_test"]["fusion"]
    bundle = {
        "models": {MODEL_NAME: components["audio_model"]},
        "best_model_name": MODEL_NAME,
        "scaler": None,
        "selector": None,
        "label_encoder": components["label_encoder"],
        "feature_names": [f"emotion2vec_{index:03d}" for index in range(768)],
        "text_model": components["text_model"],
        "fusion_audio_weight": float(components["audio_weight"]),
        "preprocessing_params": {
            "preprocessing_in_model": True,
            "feature_backend": "emotion2vec",
            "emotion2vec_model": components["emotion2vec_model"],
            "embedding_dimension": 768,
            "reliability_threshold": float(
                metrics["reliability_warning"]["selected_threshold"]["threshold"]
            ),
            "reliability_threshold_source": (
                "group-safe multimodal train OOF; warning only"
            ),
            "sample_rate": 16000,
            "denoise": False,
            "trim_silence": False,
            "normalize_amplitude": False,
            "evaluation_protocol": metrics["protocol"],
            "optional_transcript": True,
        },
        "metrics": {
            MODEL_NAME: metrics["previous_official_test"]["audio"],
            f"{MODEL_NAME} + transcript": report,
        },
        "dataset_info": {
            "audio_train_records": metrics["audio_train_records"],
            "expanded_unique_text_train_records": metrics[
                "expanded_unique_text_train_records"
            ],
            "classes": list(components["label_encoder"].classes_),
            "holdouts_used_for_selection": False,
            "deployment_limitation": metrics["deployment_limitation"],
        },
        "ensemble_weights": {
            "audio": float(components["audio_weight"]),
            "text": float(1.0 - components["audio_weight"]),
        },
        "created_at": timestamp(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
