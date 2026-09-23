from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.multimodal_components import (
    _load_embeddings,
    _select_model_and_inner_probabilities,
)
from scripts.experiment_common import CLASSES, sample_train as _sample_train
from scripts.evaluation_components import _evaluate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the train-selected audio/text late-fusion model and evaluate holdouts"
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--secondary-csv", type=Path, required=True)
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--nested-cv-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    source = pd.read_csv(args.train_csv)
    train = _sample_train(source, 3000, args.seed)
    official_test = source[
        source["dataset_split"].astype(str).str.lower().eq("test")
    ].copy().sort_values(["emotion", "file_path"]).reset_index(drop=True)
    secondary = pd.read_csv(args.secondary_csv).sort_values(
        ["emotion", "file_path"]
    ).reset_index(drop=True)
    encoder = LabelEncoder().fit(CLASSES)
    y_train = encoder.transform(train["emotion"].astype(str))
    groups_train = train["speaker_id"].astype(str).to_numpy()
    X_train = _load_embeddings(train, args.embedding_cache_dir)
    texts_train = train["speaker_text"].fillna("").astype(str).to_numpy()

    audio_model, audio_oof, audio_c, audio_scores = (
        _select_model_and_inner_probabilities(
            X_train, y_train, groups_train, args.seed, "audio"
        )
    )
    text_model, text_oof, text_c, text_scores = (
        _select_model_and_inner_probabilities(
            texts_train, y_train, groups_train, args.seed + 100, "text"
        )
    )
    weight_scores: dict[str, float] = {}
    for audio_weight in np.linspace(0.0, 1.0, 11):
        probabilities = audio_weight * audio_oof + (1.0 - audio_weight) * text_oof
        weight_scores[f"{audio_weight:.1f}"] = float(
            f1_score(y_train, np.argmax(probabilities, axis=1), average="macro")
        )
    best = max(weight_scores.values())
    audio_weight = max(
        float(key) for key, value in weight_scores.items() if value >= best - 0.002
    )

    nested_report = json.loads(args.nested_cv_metrics.read_text(encoding="utf-8"))
    official_report, official_predictions = _evaluate_manifest(
        "previous_official_test_regression_check",
        official_test,
        args.embedding_cache_dir,
        audio_model,
        text_model,
        audio_weight,
        encoder,
        args.seed,
    )
    secondary_report, secondary_predictions = _evaluate_manifest(
        "recording_disjoint_secondary_holdout_same_test_speaker_pool",
        secondary,
        args.embedding_cache_dir,
        audio_model,
        text_model,
        audio_weight,
        encoder,
        args.seed + 1,
    )
    report = {
        "protocol": "train_selected_bimodal_late_fusion_with_two_holdout_checks",
        "train_records": int(len(train)),
        "train_speakers": int(train["speaker_id"].astype(str).nunique()),
        "selection": {
            "audio_c": float(audio_c),
            "audio_inner_scores": audio_scores,
            "text_c": float(text_c),
            "text_inner_scores": text_scores,
            "fusion_audio_weight": float(audio_weight),
            "fusion_inner_scores": weight_scores,
            "holdouts_used_for_selection": False,
        },
        "nested_grouped_oof": nested_report["metrics"],
        "previous_official_test": official_report,
        "secondary_holdout": secondary_report,
        "secondary_holdout_limitation": (
            "No recording overlap with the previous 15k manifest, but most speakers "
            "belong to the same official-test speaker pool."
        ),
        "duration_sec": float(time.perf_counter() - started),
    }
    components = {
        "audio_model": audio_model,
        "text_model": text_model,
        "audio_weight": float(audio_weight),
        "label_encoder": encoder,
        "emotion2vec_model": "emotion2vec/emotion2vec_plus_base",
        "protocol": report["protocol"],
    }
    joblib.dump(components, args.output_dir / "multimodal_fusion_components.joblib")
    official_predictions.to_csv(
        args.output_dir / "official_test_multimodal_predictions.csv", index=False
    )
    secondary_predictions.to_csv(
        args.output_dir / "secondary_holdout_multimodal_predictions.csv", index=False
    )
    (args.output_dir / "multimodal_fusion_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
