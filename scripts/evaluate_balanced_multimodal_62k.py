from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluation_components import _evaluate_manifest, _metrics
from scripts.multimodal_components import _audio_model, _load_embeddings, _text_model
from scripts.experiment_common import CLASSES


def _selective(predictions: pd.DataFrame, classes: np.ndarray, threshold: float) -> dict:
    probabilities = predictions[[f"fusion_prob_{name}" for name in classes]].to_numpy()
    chosen = classes[np.argmax(probabilities, axis=1)]
    mask = np.max(probabilities, axis=1) >= threshold
    return {
        "coverage": float(mask.mean()),
        "accuracy": float((chosen[mask] == predictions["emotion"].astype(str).to_numpy()[mask]).mean()),
        "records": int(mask.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed v10 architecture after scaling both branches to balanced 62k"
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--secondary-csv", type=Path, required=True)
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audio-c", type=float, default=0.03)
    parser.add_argument("--text-c", type=float, default=0.3)
    parser.add_argument("--audio-weight", type=float, default=0.4)
    parser.add_argument("--warning-threshold", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    train = pd.read_csv(args.train_csv)
    train = train[train["dataset_split"].astype(str).eq("train")].copy()
    train = train.drop_duplicates("hash_id").reset_index(drop=True)
    comparison = pd.read_csv(args.comparison_csv)
    reference_train = comparison[
        comparison["dataset_split"].astype(str).eq("train")
    ].copy().reset_index(drop=True)
    official_test = comparison[
        comparison["dataset_split"].astype(str).eq("test")
    ].copy().reset_index(drop=True)
    secondary = pd.read_csv(args.secondary_csv).reset_index(drop=True)
    distribution = train["emotion"].value_counts().to_dict()
    if set(distribution) != set(CLASSES) or len(set(distribution.values())) != 1:
        raise ValueError(f"Training data must have equal counts in four classes: {distribution}")
    if set(train["speaker_id"].astype(str)) & set(official_test["speaker_id"].astype(str)):
        raise ValueError("Training data overlap official-test speakers")
    if set(train["hash_id"].astype(str)) & set(official_test["hash_id"].astype(str)):
        raise ValueError("Training data overlap official-test recordings")

    encoder = LabelEncoder().fit(CLASSES)
    train_y = encoder.transform(train["emotion"].astype(str))
    train_groups = train["speaker_id"].astype(str).to_numpy()
    train_texts = train["speaker_text"].fillna("").astype(str).to_numpy()
    train_X = _load_embeddings(train, args.embedding_cache_dir)
    reference_y = encoder.transform(reference_train["emotion"].astype(str))
    reference_groups = reference_train["speaker_id"].astype(str).to_numpy()
    reference_texts = reference_train["speaker_text"].fillna("").astype(str).to_numpy()
    reference_X = _load_embeddings(reference_train, args.embedding_cache_dir)
    print(
        f"Loaded {len(train)} training and {len(reference_train)} reference embeddings",
        flush=True,
    )

    audio_oof = np.zeros((len(reference_train), len(CLASSES)), dtype=float)
    text_oof = np.zeros_like(audio_oof)
    fold_details = []
    splits = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=args.seed
    ).split(reference_X, reference_y, reference_groups)
    for fold, (_, validation_index) in enumerate(splits, start=1):
        validation_speakers = set(reference_groups[validation_index])
        fit_mask = ~np.isin(train_groups, list(validation_speakers))
        if set(train_groups[fit_mask]) & validation_speakers:
            raise RuntimeError("Speaker leakage in outer fold")
        audio_model = _audio_model(args.audio_c, train_X.shape[1], args.seed)
        audio_model.fit(train_X[fit_mask], train_y[fit_mask])
        text_model = _text_model(args.text_c, args.seed + 100)
        text_model.fit(train_texts[fit_mask], train_y[fit_mask])
        audio_oof[validation_index] = audio_model.predict_proba(
            reference_X[validation_index]
        )
        text_oof[validation_index] = text_model.predict_proba(
            reference_texts[validation_index]
        )
        fold_details.append(
            {
                "fold": fold,
                "fit_records": int(fit_mask.sum()),
                "validation_records": int(len(validation_index)),
                "excluded_validation_speakers": int(len(validation_speakers)),
            }
        )
        print(f"Grouped reference fold {fold}/5 complete", flush=True)

    fused_oof = args.audio_weight * audio_oof + (1.0 - args.audio_weight) * text_oof
    grouped_oof = {
        "audio": _metrics(reference_y, audio_oof, encoder),
        "text": _metrics(reference_y, text_oof, encoder),
        "fusion": _metrics(reference_y, fused_oof, encoder),
    }
    oof_predictions = reference_train[
        ["hash_id", "emotion", "speaker_id", "speaker_text"]
    ].copy()
    oof_predictions["audio_class"] = encoder.classes_[np.argmax(audio_oof, axis=1)]
    oof_predictions["fusion_class"] = encoder.classes_[np.argmax(fused_oof, axis=1)]
    for index, name in enumerate(encoder.classes_):
        oof_predictions[f"fusion_prob_{name}"] = fused_oof[:, index]
    oof_predictions.to_csv(args.output_dir / "reference_grouped_oof.csv", index=False)

    audio_model = _audio_model(args.audio_c, train_X.shape[1], args.seed)
    audio_model.fit(train_X, train_y)
    text_model = _text_model(args.text_c, args.seed + 100)
    text_model.fit(train_texts, train_y)
    print("Final fixed models fitted on 62k", flush=True)
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
    report = {
        "protocol": "balanced_62k_both_branches_fixed_v10_hyperparameters",
        "audio_train_records": int(len(train)),
        "expanded_unique_text_train_records": int(len(train)),
        "class_distribution": {str(k): int(v) for k, v in distribution.items()},
        "annotation_confidence_minimum": 0.8,
        "selection": {
            "audio_c": float(args.audio_c),
            "text_c": float(args.text_c),
            "fusion_audio_weight": float(args.audio_weight),
            "warning_threshold": float(args.warning_threshold),
            "all_hyperparameters_fixed_from_v10": True,
        },
        "reference_grouped_oof": grouped_oof,
        "reference_grouped_oof_records": int(len(reference_train)),
        "reference_grouped_oof_fold_details": fold_details,
        "previous_official_test": official_report,
        "secondary_holdout": secondary_report,
        "reliability_warning": {
            "selection_rule": "Fixed at 0.45 from prior v10; class is not changed",
            "selected_threshold": {"threshold": float(args.warning_threshold)},
            "reference_grouped_oof": _selective(
                oof_predictions, encoder.classes_, args.warning_threshold
            ),
            "previous_official_test": _selective(
                official_predictions, encoder.classes_, args.warning_threshold
            ),
            "secondary_holdout": _selective(
                secondary_predictions, encoder.classes_, args.warning_threshold
            ),
        },
        "evaluation_limitation": (
            "Both holdouts have been inspected in previous iterations; comparison is "
            "a regression check, not a new untouched estimate."
        ),
        "deployment_limitation": (
            "Fusion requires a transcript. Audio-only prediction remains available."
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
        args.output_dir / "official_test_predictions.csv", index=False
    )
    secondary_predictions.to_csv(
        args.output_dir / "secondary_holdout_predictions.csv", index=False
    )
    (args.output_dir / "expanded_multimodal_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "reference_grouped_oof_macro_f1": grouped_oof["fusion"]["macro_f1"],
                "official_test_macro_f1": official_report["fusion"]["macro_f1"],
                "secondary_holdout_macro_f1": secondary_report["fusion"]["macro_f1"],
                "duration_sec": report["duration_sec"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
