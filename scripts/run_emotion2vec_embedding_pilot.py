from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.features import FeatureExtractor
from scripts.experiment_common import CLASSES, classification_metrics as _metrics, sample_train as _sample_train
from scripts.multimodal_components import MODEL_ID, _cache_path, _embedding_pipeline
from scripts.run_specialist_experiment import _meta_cross_fitted_router


def _extract_embeddings(data: pd.DataFrame, cache_dir: Path, batch_size: int) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = [_cache_path(cache_dir, value) for value in data["file_path"].astype(str)]
    missing = [index for index, path in enumerate(cache_paths) if not path.is_file()]
    if missing:
        logging.disable(logging.INFO)
        from funasr import AutoModel

        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                model = AutoModel(
                    model=MODEL_ID,
                    hub="hf",
                    device="cpu",
                    disable_update=True,
                    disable_pbar=True,
                    ncpu=8,
                )
        for start in range(0, len(missing), max(1, batch_size)):
            indices = missing[start : start + max(1, batch_size)]
            paths = data.iloc[indices]["file_path"].astype(str).tolist()
            with open(os.devnull, "w", encoding="utf-8") as sink:
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    generated = model.generate(
                        input=paths,
                        granularity="utterance",
                        extract_embedding=True,
                        batch_size=max(1, batch_size),
                    )
            by_key = {str(item["key"]): np.asarray(item["feats"], dtype=np.float32) for item in generated}
            for index in indices:
                key = Path(str(data.iloc[index]["file_path"])).stem
                embedding = by_key[key].reshape(-1)
                if embedding.shape != (768,):
                    raise ValueError(f"Unexpected embedding shape {embedding.shape} for {key}")
                np.save(cache_paths[index], embedding, allow_pickle=False)
            print(f"Embeddings: {min(start + batch_size, len(missing))}/{len(missing)} new", flush=True)
    return np.vstack([np.load(path, allow_pickle=False) for path in cache_paths]).astype(np.float32)


def _specialist_candidates(seed: int, feature_count: int) -> dict[str, Pipeline]:
    return {
        "logistic": _embedding_pipeline(
            LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=1200,
                solver="lbfgs",
                random_state=seed,
            ),
            feature_count,
        ),
        "hist_gradient_boosting": _embedding_pipeline(
            HistGradientBoostingClassifier(
                class_weight="balanced",
                learning_rate=0.06,
                max_iter=140,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.0,
                early_stopping=True,
                random_state=seed,
            ),
            feature_count,
        ),
        "extra_trees": _embedding_pipeline(
            ExtraTreesClassifier(
                n_estimators=250,
                max_depth=12,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            ),
            feature_count,
            scaled=False,
        ),
    }


def _fit_specialist(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    inner_splits: int,
) -> tuple[CalibratedClassifierCV, str, dict[str, float]]:
    splits = list(
        StratifiedGroupKFold(
            n_splits=inner_splits, shuffle=True, random_state=seed
        ).split(X, y, groups)
    )
    candidates = _specialist_candidates(seed, X.shape[1])
    scores = {
        name: float(
            np.mean(
                cross_val_score(
                    estimator,
                    X,
                    y,
                    cv=splits,
                    scoring="average_precision",
                    n_jobs=1,
                )
            )
        )
        for name, estimator in candidates.items()
    }
    complexity = {"logistic": 0, "hist_gradient_boosting": 1, "extra_trees": 2}
    best = max(scores.values())
    selected = min(
        [name for name, score in scores.items() if score >= best - 0.005],
        key=lambda name: complexity[name],
    )
    model = CalibratedClassifierCV(
        estimator=clone(candidates[selected]), method="sigmoid", cv=splits, ensemble=True
    )
    model.fit(X, y)
    return model, selected, scores


def _fit_embedding_baseline(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    inner_splits: int,
) -> tuple[Pipeline, float, dict[str, float]]:
    splits = list(
        StratifiedGroupKFold(
            n_splits=inner_splits, shuffle=True, random_state=seed
        ).split(X, y, groups)
    )
    scores: dict[str, float] = {}
    models: dict[float, Pipeline] = {}
    for c_value in [0.01, 0.03, 0.1, 0.3]:
        model = _embedding_pipeline(
            LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=1500,
                solver="lbfgs",
                random_state=seed,
            ),
            X.shape[1],
        )
        models[c_value] = model
        values = cross_val_score(
            model, X, y, cv=splits, scoring="f1_macro", n_jobs=1
        )
        scores[str(c_value)] = float(np.mean(values))
    best = max(scores.values())
    selected_c = min(float(key) for key, value in scores.items() if value >= best - 0.005)
    selected_model = models[selected_c]
    selected_model.fit(X, y)
    return selected_model, selected_c, scores


def _hand_pipeline(seed: int, feature_count: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(f_classif, k=min(96, feature_count))),
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=1200,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-only emotion2vec embedding specialist pilot")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-class", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(args.csv)
    data = _sample_train(source, args.max_per_class, args.seed)
    test_speakers = set(
        source.loc[source["dataset_split"].astype(str).str.lower() == "test", "speaker_id"].astype(str)
    )
    if set(data["speaker_id"].astype(str)) & test_speakers:
        raise RuntimeError("Pilot intersects official test speakers.")
    started = time.perf_counter()
    embeddings = _extract_embeddings(data, args.embedding_cache_dir, args.batch_size)

    feature_extractor = FeatureExtractor(
        args.feature_cache_dir,
        sample_rate=16000,
        denoise=False,
        trim=True,
        normalize=True,
    )
    hand = feature_extractor.build_matrix(data)
    if hand.errors:
        raise RuntimeError(f"Hand feature errors: {hand.errors[:3]}")
    hand_features = hand.X
    groups = np.asarray(data["speaker_id"].astype(str))
    encoder = LabelEncoder().fit(CLASSES)
    y = encoder.transform(data["emotion"].astype(str))
    n = len(y)

    hand_probabilities = np.zeros((n, len(CLASSES)), dtype=float)
    embedding_probabilities = np.zeros((n, len(CLASSES)), dtype=float)
    specialist_probabilities = np.zeros((n, len(CLASSES)), dtype=float)
    pairwise_calm_probability = np.zeros(n, dtype=float)
    selections: list[dict[str, Any]] = []
    outer = StratifiedGroupKFold(
        n_splits=args.outer_splits, shuffle=True, random_state=args.seed
    )
    for fold, (train_idx, validation_idx) in enumerate(outer.split(embeddings, y, groups)):
        hand_model = _hand_pipeline(args.seed + fold, hand_features.shape[1])
        hand_model.fit(hand_features[train_idx], y[train_idx])
        hand_probabilities[validation_idx] = hand_model.predict_proba(hand_features[validation_idx])

        embedding_model, selected_c, scores = _fit_embedding_baseline(
            embeddings[train_idx],
            y[train_idx],
            groups[train_idx],
            args.seed + fold,
            args.inner_splits,
        )
        embedding_probabilities[validation_idx] = embedding_model.predict_proba(
            embeddings[validation_idx]
        )
        selections.append(
            {
                "outer_fold": fold,
                "task": "embedding_multiclass",
                "selected": f"logistic_C_{selected_c}",
                "inner_macro_f1": scores,
            }
        )

        for class_index, label in enumerate(CLASSES):
            binary = (y[train_idx] == class_index).astype(int)
            model, selected, ap_scores = _fit_specialist(
                embeddings[train_idx],
                binary,
                groups[train_idx],
                args.seed + 100 + fold * 10 + class_index,
                args.inner_splits,
            )
            specialist_probabilities[validation_idx, class_index] = model.predict_proba(
                embeddings[validation_idx]
            )[:, 1]
            selections.append(
                {
                    "outer_fold": fold,
                    "task": f"{label}_vs_rest",
                    "selected": selected,
                    "inner_average_precision": ap_scores,
                }
            )

        pair_mask = np.isin(y[train_idx], [CLASSES.index("calm"), CLASSES.index("sadness")])
        pair_idx = train_idx[pair_mask]
        pair_y = (y[pair_idx] == CLASSES.index("calm")).astype(int)
        pair_model, selected, ap_scores = _fit_specialist(
            embeddings[pair_idx],
            pair_y,
            groups[pair_idx],
            args.seed + 500 + fold,
            args.inner_splits,
        )
        pairwise_calm_probability[validation_idx] = pair_model.predict_proba(
            embeddings[validation_idx]
        )[:, 1]
        selections.append(
            {
                "outer_fold": fold,
                "task": "calm_vs_sadness",
                "selected": selected,
                "inner_average_precision": ap_scores,
            }
        )
        print(f"CV fold {fold + 1}/{args.outer_splits} complete", flush=True)

    hand_pred = np.argmax(hand_probabilities, axis=1)
    embedding_pred = np.argmax(embedding_probabilities, axis=1)
    routed_pred, accepted, router_settings = _meta_cross_fitted_router(
        y,
        groups,
        embedding_pred,
        specialist_probabilities,
        pairwise_calm_probability,
        args.seed,
    )
    changed = accepted & (routed_pred != embedding_pred)
    results = {
        "protocol": "official_train_only_emotion2vec_frozen_embedding_nested_grouped_pilot",
        "model": MODEL_ID,
        "records": int(n),
        "speakers": int(len(np.unique(groups))),
        "official_test_used": False,
        "hand_feature_baseline": _metrics(y, hand_pred),
        "embedding_multiclass": _metrics(y, embedding_pred),
        "embedding_specialist_router": _metrics(y, routed_pred, changed),
        "embedding_gain_over_hand_features": float(
            f1_score(y, embedding_pred, average="macro")
            - f1_score(y, hand_pred, average="macro")
        ),
        "router_gain_over_embedding_multiclass": float(
            f1_score(y, routed_pred, average="macro")
            - f1_score(y, embedding_pred, average="macro")
        ),
        "specialist_average_precision": {
            label: float(
                average_precision_score(
                    (y == index).astype(int), specialist_probabilities[:, index]
                )
            )
            for index, label in enumerate(CLASSES)
        },
        "selected_model_counts": dict(Counter(item["selected"] for item in selections)),
        "router_settings": router_settings,
        "model_selections": selections,
        "duration_sec": float(time.perf_counter() - started),
    }
    predictions = data[["file_path", "emotion", "speaker_id"]].copy()
    predictions["hand_class"] = encoder.inverse_transform(hand_pred)
    predictions["embedding_class"] = encoder.inverse_transform(embedding_pred)
    predictions["router_class"] = encoder.inverse_transform(routed_pred)
    predictions["router_changed"] = changed
    for index, label in enumerate(CLASSES):
        predictions[f"embedding_prob_{label}"] = embedding_probabilities[:, index]
        predictions[f"specialist_prob_{label}"] = specialist_probabilities[:, index]
    predictions.to_csv(args.output_dir / "embedding_pilot_oof_predictions.csv", index=False)
    (args.output_dir / "embedding_pilot_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
