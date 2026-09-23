from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluation_components import _evaluate_manifest
from scripts.multimodal_components import (
    _load_embeddings,
    _select_model_and_inner_probabilities,
    _text_model,
)
from scripts.experiment_common import CLASSES, sample_train as _sample_train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize audio plus expanded-transcript late fusion"
    )
    parser.add_argument("--audio-csv", type=Path, required=True)
    parser.add_argument("--expanded-text-csv", type=Path, required=True)
    parser.add_argument("--secondary-csv", type=Path, required=True)
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--expanded-oof-metrics", type=Path, required=True)
    parser.add_argument("--expanded-oof-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audio-weight", type=float, default=0.4)
    parser.add_argument("--text-c", type=float, default=0.3)
    parser.add_argument("--no-union-audio-train", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    source = pd.read_csv(args.audio_csv)
    audio_train = _sample_train(source, 3000, args.seed)
    official_test = source[
        source["dataset_split"].astype(str).str.lower().eq("test")
    ].copy().sort_values(["emotion", "file_path"]).reset_index(drop=True)
    secondary = pd.read_csv(args.secondary_csv).sort_values(
        ["emotion", "file_path"]
    ).reset_index(drop=True)
    expanded = pd.read_csv(args.expanded_text_csv)
    expanded = expanded[
        expanded["dataset_split"].astype(str).str.lower().eq("train")
    ].copy()
    if not args.no_union_audio_train:
        expanded = pd.concat([expanded, audio_train], ignore_index=True)
    expanded = expanded.drop_duplicates("hash_id").reset_index(drop=True)

    encoder = LabelEncoder().fit(CLASSES)
    audio_y = encoder.transform(audio_train["emotion"].astype(str))
    audio_groups = audio_train["speaker_id"].astype(str).to_numpy()
    audio_X = _load_embeddings(audio_train, args.embedding_cache_dir)
    audio_model, _, audio_c, audio_scores = _select_model_and_inner_probabilities(
        audio_X, audio_y, audio_groups, args.seed, "audio"
    )
    text_y = encoder.transform(expanded["emotion"].astype(str))
    text_model = _text_model(args.text_c, args.seed + 100)
    text_model.fit(
        expanded["speaker_text"].fillna("").astype(str).to_numpy(), text_y
    )

    official_report, official_predictions = _evaluate_manifest(
        "previous_official_test_regression_check",
        official_test,
        args.embedding_cache_dir,
        audio_model,
        text_model,
        args.audio_weight,
        encoder,
        args.seed,
    )
    secondary_report, secondary_predictions = _evaluate_manifest(
        "recording_disjoint_secondary_holdout_same_test_speaker_pool",
        secondary,
        args.embedding_cache_dir,
        audio_model,
        text_model,
        args.audio_weight,
        encoder,
        args.seed + 1,
    )
    oof = json.loads(args.expanded_oof_metrics.read_text(encoding="utf-8"))
    oof_predictions = pd.read_csv(args.expanded_oof_predictions)
    oof_probability_columns = [
        f"audio_expanded_text_fusion_prob_{label}" for label in encoder.classes_
    ]
    oof_probabilities = oof_predictions[oof_probability_columns].to_numpy()
    oof_truth = oof_predictions["emotion"].astype(str).to_numpy()
    oof_classes = encoder.classes_[np.argmax(oof_probabilities, axis=1)]
    oof_confidence = np.max(oof_probabilities, axis=1)
    threshold_candidates = np.arange(0.35, 0.86, 0.01)
    threshold_rows = []
    for threshold in threshold_candidates:
        mask = oof_confidence >= threshold
        threshold_rows.append(
            {
                "threshold": float(round(threshold, 2)),
                "coverage": float(mask.mean()),
                "accuracy": float((oof_classes[mask] == oof_truth[mask]).mean()),
                "records": int(mask.sum()),
            }
        )
    eligible = [row for row in threshold_rows if row["accuracy"] >= 0.84]
    selected_threshold = min(eligible, key=lambda row: row["threshold"])

    def selective_metrics(predictions: pd.DataFrame) -> dict:
        columns = [f"fusion_prob_{label}" for label in encoder.classes_]
        probabilities = predictions[columns].to_numpy()
        confidence = np.max(probabilities, axis=1)
        predicted = encoder.classes_[np.argmax(probabilities, axis=1)]
        truth = predictions["emotion"].astype(str).to_numpy()
        mask = confidence >= selected_threshold["threshold"]
        return {
            "coverage": float(mask.mean()),
            "accuracy": float((predicted[mask] == truth[mask]).mean()),
            "records": int(mask.sum()),
        }
    report = {
        "protocol": "audio_12k_plus_group_safe_expanded_text_59k_late_fusion",
        "audio_train_records": int(len(audio_train)),
        "expanded_unique_text_train_records": int(len(expanded)),
        "selection": {
            "audio_c": float(audio_c),
            "audio_inner_scores": audio_scores,
            "text_c": float(args.text_c),
            "fusion_audio_weight": float(args.audio_weight),
            "text_c_and_weight_fixed_from_prior_nested_cv": True,
            "holdouts_used_for_selection": False,
        },
        "grouped_oof": oof["metrics"],
        "previous_official_test": official_report,
        "secondary_holdout": secondary_report,
        "reliability_warning": {
            "selection_rule": (
                "Lowest 0.01-grid confidence threshold reaching at least 0.84 "
                "selective accuracy on group-safe train OOF. The threshold only "
                "adds a warning and never changes the predicted class."
            ),
            "selected_threshold": selected_threshold,
            "previous_official_test": selective_metrics(official_predictions),
            "secondary_holdout": selective_metrics(secondary_predictions),
        },
        "secondary_holdout_limitation": (
            "No recording overlap with the previous 15k manifest, but most speakers "
            "belong to the same official-test speaker pool."
        ),
        "deployment_limitation": (
            "Fusion requires a transcript. The existing audio-only app remains the "
            "fallback when transcript text is unavailable."
        ),
        "duration_sec": float(time.perf_counter() - started),
    }
    components = {
        "audio_model": audio_model,
        "text_model": text_model,
        "audio_weight": float(args.audio_weight),
        "label_encoder": encoder,
        "emotion2vec_model": "emotion2vec/emotion2vec_plus_base",
        "protocol": report["protocol"],
    }
    joblib.dump(components, args.output_dir / "expanded_multimodal_components.joblib")
    official_predictions.to_csv(
        args.output_dir / "official_test_expanded_multimodal_predictions.csv",
        index=False,
    )
    secondary_predictions.to_csv(
        args.output_dir / "secondary_holdout_expanded_multimodal_predictions.csv",
        index=False,
    )
    (args.output_dir / "expanded_multimodal_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
