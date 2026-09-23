from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.multimodal_components import C_VALUES, _audio_model, _load_embeddings, _select_model_and_inner_probabilities, _splits, _text_model
from scripts.experiment_common import CLASSES, classification_metrics as _metrics, sample_train as _sample_train


def _select_binary_audio(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[Pipeline, float]:
    splits = _splits(X, y, groups, seed)
    scored: list[tuple[float, float, Pipeline]] = []
    for c_value in C_VALUES:
        model = _audio_model(c_value, X.shape[1], seed)
        score = float(
            np.mean(
                cross_val_score(
                    model, X, y, cv=splits, scoring="f1_macro", n_jobs=1
                )
            )
        )
        scored.append((score, c_value, model))
    best = max(item[0] for item in scored)
    _, selected_c, model = min(
        [item for item in scored if item[0] >= best - 0.005],
        key=lambda item: item[1],
    )
    model.fit(X, y)
    return model, float(selected_c)


def _hierarchical_probabilities(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_validation: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    anger = CLASSES.index("anger")
    calm = CLASSES.index("calm")
    joy = CLASSES.index("joy")
    sadness = CLASSES.index("sadness")

    high_y = np.isin(y_train, [anger, joy]).astype(int)
    arousal_model, arousal_c = _select_binary_audio(
        X_train, high_y, groups_train, seed
    )
    p_high = arousal_model.predict_proba(X_validation)[:, 1]

    high_mask = np.isin(y_train, [anger, joy])
    high_model, high_c = _select_binary_audio(
        X_train[high_mask],
        (y_train[high_mask] == joy).astype(int),
        groups_train[high_mask],
        seed + 1,
    )
    p_joy_given_high = high_model.predict_proba(X_validation)[:, 1]

    low_mask = np.isin(y_train, [calm, sadness])
    low_model, low_c = _select_binary_audio(
        X_train[low_mask],
        (y_train[low_mask] == sadness).astype(int),
        groups_train[low_mask],
        seed + 2,
    )
    p_sad_given_low = low_model.predict_proba(X_validation)[:, 1]

    probabilities = np.zeros((len(X_validation), len(CLASSES)), dtype=float)
    probabilities[:, anger] = p_high * (1.0 - p_joy_given_high)
    probabilities[:, joy] = p_high * p_joy_given_high
    probabilities[:, calm] = (1.0 - p_high) * (1.0 - p_sad_given_low)
    probabilities[:, sadness] = (1.0 - p_high) * p_sad_given_low
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities, {
        "arousal_c": arousal_c,
        "anger_joy_c": high_c,
        "calm_sadness_c": low_c,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train-only pilot for hierarchical audio and transcript fusion"
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-class", type=int, default=250)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(args.csv)
    data = _sample_train(source, args.max_per_class, args.seed)
    embeddings = _load_embeddings(data, args.embedding_cache_dir)
    texts = data["speaker_text"].fillna("").astype(str).to_numpy()
    groups = data["speaker_id"].astype(str).to_numpy()
    encoder = LabelEncoder().fit(CLASSES)
    y = encoder.transform(data["emotion"].astype(str))
    n = len(y)

    audio_probabilities = np.zeros((n, len(CLASSES)), dtype=float)
    hierarchy_probabilities = np.zeros_like(audio_probabilities)
    text_probabilities = np.zeros_like(audio_probabilities)
    fusion_probabilities = np.zeros_like(audio_probabilities)
    selections: list[dict[str, Any]] = []
    started = time.perf_counter()
    outer = StratifiedGroupKFold(
        n_splits=args.outer_splits, shuffle=True, random_state=args.seed
    )
    for fold, (train_idx, validation_idx) in enumerate(
        outer.split(embeddings, y, groups)
    ):
        audio_model, audio_inner, audio_c, audio_scores = (
            _select_model_and_inner_probabilities(
                embeddings[train_idx],
                y[train_idx],
                groups[train_idx],
                args.seed + fold,
                "audio",
            )
        )
        text_model, text_inner, text_c, text_scores = (
            _select_model_and_inner_probabilities(
                texts[train_idx],
                y[train_idx],
                groups[train_idx],
                args.seed + 100 + fold,
                "text",
            )
        )
        audio_probabilities[validation_idx] = audio_model.predict_proba(
            embeddings[validation_idx]
        )
        text_probabilities[validation_idx] = text_model.predict_proba(
            texts[validation_idx]
        )

        weight_scores: dict[str, float] = {}
        for audio_weight in np.linspace(0.0, 1.0, 11):
            combined = audio_weight * audio_inner + (1.0 - audio_weight) * text_inner
            score = f1_score(y[train_idx], np.argmax(combined, axis=1), average="macro")
            weight_scores[f"{audio_weight:.1f}"] = float(score)
        best_weight_score = max(weight_scores.values())
        selected_weight = max(
            float(key)
            for key, value in weight_scores.items()
            if value >= best_weight_score - 0.002
        )
        fusion_probabilities[validation_idx] = (
            selected_weight * audio_probabilities[validation_idx]
            + (1.0 - selected_weight) * text_probabilities[validation_idx]
        )
        hierarchy_probabilities[validation_idx], hierarchy_config = (
            _hierarchical_probabilities(
                embeddings[train_idx],
                y[train_idx],
                groups[train_idx],
                embeddings[validation_idx],
                args.seed + 200 + fold * 10,
            )
        )
        selections.append(
            {
                "fold": fold,
                "audio_c": audio_c,
                "audio_inner_scores": audio_scores,
                "text_c": text_c,
                "text_inner_scores": text_scores,
                "fusion_audio_weight": selected_weight,
                "fusion_inner_scores": weight_scores,
                "hierarchy": hierarchy_config,
            }
        )
        print(f"Fold {fold + 1}/{args.outer_splits} complete", flush=True)

    variants = {
        "audio_flat": audio_probabilities,
        "audio_hierarchy": hierarchy_probabilities,
        "text_only": text_probabilities,
        "audio_text_late_fusion": fusion_probabilities,
    }
    metrics = {
        name: _metrics(y, np.argmax(probabilities, axis=1))
        for name, probabilities in variants.items()
    }
    result = {
        "protocol": "official_train_only_nested_speaker_grouped_context_hierarchy_pilot",
        "records": int(n),
        "speakers": int(len(np.unique(groups))),
        "official_test_used": False,
        "text_unique": int(pd.Series(texts).nunique()),
        "metrics": metrics,
        "gain_over_audio": {
            name: float(value["macro_f1"] - metrics["audio_flat"]["macro_f1"])
            for name, value in metrics.items()
            if name != "audio_flat"
        },
        "selection_counts": {
            "audio_c": dict(Counter(str(item["audio_c"]) for item in selections)),
            "text_c": dict(Counter(str(item["text_c"]) for item in selections)),
            "fusion_audio_weight": dict(
                Counter(str(item["fusion_audio_weight"]) for item in selections)
            ),
        },
        "fold_selections": selections,
        "duration_sec": float(time.perf_counter() - started),
    }
    predictions = data[["file_path", "emotion", "speaker_id", "speaker_text"]].copy()
    for name, probabilities in variants.items():
        predictions[f"{name}_class"] = encoder.inverse_transform(
            np.argmax(probabilities, axis=1)
        )
    predictions.to_csv(args.output_dir / "context_hierarchy_oof_predictions.csv", index=False)
    (args.output_dir / "context_hierarchy_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
