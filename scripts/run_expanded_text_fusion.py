from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.multimodal_components import (
    _load_embeddings,
    _select_model_and_inner_probabilities,
    _text_model,
)
from scripts.experiment_common import CLASSES, classification_metrics as _metrics, sample_train as _sample_train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group-safe OOF pilot using additional transcript-only training data"
    )
    parser.add_argument("--audio-csv", type=Path, required=True)
    parser.add_argument("--expanded-text-csv", type=Path, required=True)
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audio-weight", type=float, default=0.4)
    parser.add_argument("--text-c", type=float, default=0.3)
    parser.add_argument("--no-union-evaluation", action="store_true")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(args.audio_csv)
    evaluation = _sample_train(source, 3000, args.seed)
    expanded = pd.read_csv(args.expanded_text_csv)
    expanded = expanded[
        expanded["dataset_split"].astype(str).str.lower().eq("train")
    ].copy()
    # Union ensures every ordinary outer-training record remains available even if
    # the larger balanced sample happened not to select it. Speaker exclusion below
    # is the leakage barrier.
    if not args.no_union_evaluation:
        expanded = pd.concat([expanded, evaluation], ignore_index=True)
    expanded = expanded.drop_duplicates("hash_id").reset_index(drop=True)

    embeddings = _load_embeddings(evaluation, args.embedding_cache_dir)
    texts = evaluation["speaker_text"].fillna("").astype(str).to_numpy()
    groups = evaluation["speaker_id"].astype(str).to_numpy()
    encoder = LabelEncoder().fit(CLASSES)
    y = encoder.transform(evaluation["emotion"].astype(str))

    audio_probabilities = np.zeros((len(evaluation), len(CLASSES)), dtype=float)
    text_probabilities = np.zeros_like(audio_probabilities)
    fold_details: list[dict] = []
    started = time.perf_counter()
    splitter = StratifiedGroupKFold(
        n_splits=args.outer_splits, shuffle=True, random_state=args.seed
    )
    for fold, (train_idx, validation_idx) in enumerate(
        splitter.split(embeddings, y, groups)
    ):
        audio_model, _, audio_c, audio_scores = _select_model_and_inner_probabilities(
            embeddings[train_idx],
            y[train_idx],
            groups[train_idx],
            args.seed + fold,
            "audio",
        )
        audio_probabilities[validation_idx] = audio_model.predict_proba(
            embeddings[validation_idx]
        )

        validation_speakers = set(groups[validation_idx])
        text_train = expanded[
            ~expanded["speaker_id"].astype(str).isin(validation_speakers)
        ]
        text_y = encoder.transform(text_train["emotion"].astype(str))
        text_model = _text_model(args.text_c, args.seed + 100 + fold)
        text_model.fit(
            text_train["speaker_text"].fillna("").astype(str).to_numpy(), text_y
        )
        text_probabilities[validation_idx] = text_model.predict_proba(
            texts[validation_idx]
        )
        fold_details.append(
            {
                "fold": fold,
                "audio_c": float(audio_c),
                "audio_inner_scores": audio_scores,
                "text_train_records": int(len(text_train)),
                "validation_speakers_excluded": int(len(validation_speakers)),
            }
        )
        print(f"Fold {fold + 1}/{args.outer_splits} complete", flush=True)

    fusion_probabilities = (
        float(args.audio_weight) * audio_probabilities
        + (1.0 - float(args.audio_weight)) * text_probabilities
    )
    variants = {
        "audio": audio_probabilities,
        "expanded_text": text_probabilities,
        "audio_expanded_text_fusion": fusion_probabilities,
    }
    metrics = {
        name: _metrics(y, np.argmax(probabilities, axis=1))
        for name, probabilities in variants.items()
    }
    result = {
        "protocol": "official_train_only_grouped_oof_with_expanded_transcript_pool",
        "evaluation_records": int(len(evaluation)),
        "expanded_unique_train_records": int(len(expanded)),
        "official_test_used": False,
        "speaker_leakage_prevention": (
            "All records from validation speakers are removed from the expanded text pool per fold."
        ),
        "fixed_from_prior_nested_cv": {
            "audio_weight": float(args.audio_weight),
            "text_c": float(args.text_c),
        },
        "metrics": metrics,
        "fusion_gain_over_audio": float(
            metrics["audio_expanded_text_fusion"]["macro_f1"]
            - metrics["audio"]["macro_f1"]
        ),
        "fold_details": fold_details,
        "duration_sec": float(time.perf_counter() - started),
    }
    predictions = evaluation[
        ["file_path", "emotion", "speaker_id", "speaker_text", "hash_id"]
    ].copy()
    for name, probabilities in variants.items():
        predictions[f"{name}_class"] = encoder.inverse_transform(
            np.argmax(probabilities, axis=1)
        )
        for class_index, label in enumerate(encoder.classes_):
            predictions[f"{name}_prob_{label}"] = probabilities[:, class_index]
    predictions.to_csv(args.output_dir / "expanded_text_fusion_oof.csv", index=False)
    (args.output_dir / "expanded_text_fusion_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
