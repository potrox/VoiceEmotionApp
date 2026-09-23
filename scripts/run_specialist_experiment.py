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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.features import FeatureExtractor
from scripts.experiment_common import CLASSES, classification_metrics as _metrics, sample_train as _sample_train


def _pipeline(model: Any, scaled: bool = True) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scaled:
        steps.append(("scaler", StandardScaler()))
    steps.extend(
        [
            ("selector", SelectKBest(f_classif, k=96)),
            ("model", model),
        ]
    )
    return Pipeline(steps)


def _candidates(seed: int) -> dict[str, Pipeline]:
    return {
        "logistic": _pipeline(
            LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=1200,
                solver="lbfgs",
                random_state=seed,
            )
        ),
        "hist_gradient_boosting": _pipeline(
            HistGradientBoostingClassifier(
                class_weight="balanced",
                learning_rate=0.06,
                max_iter=160,
                max_leaf_nodes=15,
                min_samples_leaf=30,
                l2_regularization=0.5,
                early_stopping=True,
                random_state=seed,
            )
        ),
        "extra_trees": _pipeline(
            ExtraTreesClassifier(
                n_estimators=250,
                max_depth=14,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            ),
            scaled=False,
        ),
    }


def _group_cv(y: np.ndarray, groups: np.ndarray, seed: int, splits: int) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed)


def _fit_best_calibrated(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    inner_splits: int,
) -> tuple[CalibratedClassifierCV, str, dict[str, float]]:
    splitter = _group_cv(y, groups, seed, inner_splits)
    splits = list(splitter.split(X, y, groups))
    candidate_scores: dict[str, float] = {}
    candidates = _candidates(seed)
    for name, estimator in candidates.items():
        values = cross_val_score(
            estimator,
            X,
            y,
            cv=splits,
            scoring="average_precision",
            n_jobs=1,
        )
        candidate_scores[name] = float(np.mean(values))
    complexity = {"logistic": 0, "hist_gradient_boosting": 1, "extra_trees": 2}
    best_score = max(candidate_scores.values())
    eligible = [name for name, score in candidate_scores.items() if score >= best_score - 0.005]
    best_name = min(eligible, key=lambda name: complexity[name])
    calibrated = CalibratedClassifierCV(
        estimator=clone(candidates[best_name]),
        method="sigmoid",
        cv=splits,
        ensemble=True,
    )
    calibrated.fit(X, y)
    return calibrated, best_name, candidate_scores


def _threshold_for_precision(
    y_true: np.ndarray,
    probability: np.ndarray,
    target_precision: float = 0.70,
    minimum_records: int = 25,
) -> float:
    fallback: tuple[float, float] | None = None
    feasible: list[tuple[float, float, float]] = []
    for threshold in np.linspace(0.35, 0.95, 25):
        accepted = probability >= threshold
        count = int(np.sum(accepted))
        if count < minimum_records:
            continue
        precision = float(np.mean(y_true[accepted] == 1))
        recall = float(np.sum((y_true == 1) & accepted) / max(np.sum(y_true == 1), 1))
        if precision >= target_precision:
            feasible.append((recall, precision, float(threshold)))
        beta_sq = 0.25
        f_half = (1 + beta_sq) * precision * recall / max(beta_sq * precision + recall, 1e-12)
        if fallback is None or f_half > fallback[0]:
            fallback = (f_half, float(threshold))
    if feasible:
        return max(feasible)[2]
    return fallback[1] if fallback else 0.95


def _route_ovr(
    baseline_pred: np.ndarray,
    specialist_probabilities: np.ndarray,
    thresholds: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    top = np.argmax(specialist_probabilities, axis=1)
    ordered = np.sort(specialist_probabilities, axis=1)
    top_probability = specialist_probabilities[np.arange(len(top)), top]
    accepted = (top_probability >= thresholds[top]) & ((ordered[:, -1] - ordered[:, -2]) >= margin)
    output = baseline_pred.copy()
    output[accepted] = top[accepted]
    return output, accepted


def _route_pairwise(
    baseline_pred: np.ndarray,
    calm_probability: np.ndarray,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    output = baseline_pred.copy()
    in_pair = np.isin(baseline_pred, [CLASSES.index("calm"), CLASSES.index("sadness")])
    confidence = np.maximum(calm_probability, 1.0 - calm_probability)
    accepted = in_pair & (confidence >= confidence_threshold)
    output[accepted] = np.where(
        calm_probability[accepted] >= 0.5,
        CLASSES.index("calm"),
        CLASSES.index("sadness"),
    )
    return output, accepted


def _meta_cross_fitted_router(
    y: np.ndarray,
    groups: np.ndarray,
    baseline_pred: np.ndarray,
    specialist_probabilities: np.ndarray,
    pairwise_calm_probability: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    routed = np.full_like(y, -1)
    accepted_all = np.zeros(len(y), dtype=bool)
    settings: list[dict[str, Any]] = []
    meta_cv = _group_cv(y, groups, seed + 1000, 5)
    for fold, (train_idx, validation_idx) in enumerate(meta_cv.split(specialist_probabilities, y, groups)):
        thresholds = np.asarray(
            [
                _threshold_for_precision(
                    (y[train_idx] == class_index).astype(int),
                    specialist_probabilities[train_idx, class_index],
                )
                for class_index in range(len(CLASSES))
            ]
        )
        best_margin = 1.0
        best_score = float(f1_score(y[train_idx], baseline_pred[train_idx], average="macro"))
        for margin in [0.0, 0.05, 0.10, 0.15, 0.20]:
            candidate, _ = _route_ovr(
                baseline_pred[train_idx], specialist_probabilities[train_idx], thresholds, margin
            )
            score = float(f1_score(y[train_idx], candidate, average="macro"))
            if score > best_score + 1e-6:
                best_score, best_margin = score, margin

        pair_threshold: float | None = None
        pair_base, _ = _route_ovr(
            baseline_pred[train_idx], specialist_probabilities[train_idx], thresholds, best_margin
        )
        for threshold in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            candidate, _ = _route_pairwise(
                pair_base, pairwise_calm_probability[train_idx], threshold
            )
            score = float(f1_score(y[train_idx], candidate, average="macro"))
            if score > best_score + 1e-6:
                best_score, pair_threshold = score, threshold

        validation_pred, accepted = _route_ovr(
            baseline_pred[validation_idx],
            specialist_probabilities[validation_idx],
            thresholds,
            best_margin,
        )
        if pair_threshold is not None:
            validation_pred, pair_accepted = _route_pairwise(
                validation_pred,
                pairwise_calm_probability[validation_idx],
                pair_threshold,
            )
            accepted |= pair_accepted
        routed[validation_idx] = validation_pred
        accepted_all[validation_idx] = accepted
        settings.append(
            {
                "fold": fold,
                "thresholds": {CLASSES[i]: float(value) for i, value in enumerate(thresholds)},
                "margin": float(best_margin),
                "pair_threshold": pair_threshold,
            }
        )
    return routed, accepted_all, settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested grouped class-specialist experiment")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--max-per-class", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(args.csv)
    data = _sample_train(source, args.max_per_class, args.seed)
    official_test_speakers = set(
        source.loc[source["dataset_split"].astype(str).str.lower() == "test", "speaker_id"].astype(str)
    )
    if set(data["speaker_id"].astype(str)) & official_test_speakers:
        raise RuntimeError("Development data intersects official test speakers.")
    extractor = FeatureExtractor(
        args.cache_dir,
        sample_rate=16000,
        denoise=False,
        trim=True,
        normalize=True,
    )
    started = time.perf_counter()

    def progress(done: int, total: int) -> None:
        if done % 500 == 0 or done == total:
            print(f"Features: {done}/{total}", flush=True)

    matrix = extractor.build_matrix(data, progress_callback=progress)
    X = matrix.X
    groups = np.asarray(matrix.speaker_ids, dtype=str)
    encoder = LabelEncoder().fit(CLASSES)
    y = encoder.transform(matrix.y)
    if list(encoder.classes_) != CLASSES:
        raise RuntimeError(f"Unexpected class order: {list(encoder.classes_)}")

    n = len(y)
    baseline_probabilities = np.zeros((n, len(CLASSES)), dtype=float)
    specialist_probabilities = np.zeros((n, len(CLASSES)), dtype=float)
    pairwise_calm_probability = np.zeros(n, dtype=float)
    selections: list[dict[str, Any]] = []
    outer = _group_cv(y, groups, args.seed, args.outer_splits)
    for fold, (train_idx, validation_idx) in enumerate(outer.split(X, y, groups)):
        baseline = _pipeline(
            LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=1200,
                solver="lbfgs",
                random_state=args.seed + fold,
            )
        )
        baseline.fit(X[train_idx], y[train_idx])
        baseline_probabilities[validation_idx] = baseline.predict_proba(X[validation_idx])

        for class_index, label in enumerate(CLASSES):
            binary_y = (y[train_idx] == class_index).astype(int)
            model, selected, scores = _fit_best_calibrated(
                X[train_idx],
                binary_y,
                groups[train_idx],
                args.seed + fold * 10 + class_index,
                args.inner_splits,
            )
            specialist_probabilities[validation_idx, class_index] = model.predict_proba(
                X[validation_idx]
            )[:, 1]
            selections.append(
                {
                    "outer_fold": fold,
                    "task": f"{label}_vs_rest",
                    "selected": selected,
                    "inner_average_precision": scores,
                }
            )
            print(f"Fold {fold + 1}: {label} -> {selected}", flush=True)

        pair_train_mask = np.isin(y[train_idx], [CLASSES.index("calm"), CLASSES.index("sadness")])
        pair_train_idx = train_idx[pair_train_mask]
        pair_y = (y[pair_train_idx] == CLASSES.index("calm")).astype(int)
        pair_model, selected, scores = _fit_best_calibrated(
            X[pair_train_idx],
            pair_y,
            groups[pair_train_idx],
            args.seed + 500 + fold,
            args.inner_splits,
        )
        pairwise_calm_probability[validation_idx] = pair_model.predict_proba(
            X[validation_idx]
        )[:, 1]
        selections.append(
            {
                "outer_fold": fold,
                "task": "calm_vs_sadness",
                "selected": selected,
                "inner_average_precision": scores,
            }
        )
        print(f"Fold {fold + 1}: calm_vs_sadness -> {selected}", flush=True)

    baseline_pred = np.argmax(baseline_probabilities, axis=1)
    routed_pred, accepted, router_settings = _meta_cross_fitted_router(
        y,
        groups,
        baseline_pred,
        specialist_probabilities,
        pairwise_calm_probability,
        args.seed,
    )
    specialist_ap = {
        label: float(average_precision_score((y == index).astype(int), specialist_probabilities[:, index]))
        for index, label in enumerate(CLASSES)
    }
    changed = accepted & (routed_pred != baseline_pred)
    results = {
        "protocol": "official_train_only_nested_speaker_grouped_specialists_meta_cross_fitted_router",
        "records": int(n),
        "speakers": int(len(np.unique(groups))),
        "outer_splits": args.outer_splits,
        "inner_splits": args.inner_splits,
        "official_test_used": False,
        "baseline": _metrics(y, baseline_pred),
        "router": _metrics(y, routed_pred, changed),
        "macro_f1_gain": float(
            f1_score(y, routed_pred, average="macro")
            - f1_score(y, baseline_pred, average="macro")
        ),
        "specialist_average_precision": specialist_ap,
        "selected_model_counts": dict(Counter(item["selected"] for item in selections)),
        "router_settings": router_settings,
        "model_selections": selections,
        "duration_sec": float(time.perf_counter() - started),
    }
    predictions = data[["file_path", "emotion", "speaker_id"]].copy()
    predictions["baseline_class"] = encoder.inverse_transform(baseline_pred)
    predictions["router_class"] = encoder.inverse_transform(routed_pred)
    predictions["router_changed"] = changed
    for index, label in enumerate(CLASSES):
        predictions[f"baseline_prob_{label}"] = baseline_probabilities[:, index]
        predictions[f"specialist_prob_{label}"] = specialist_probabilities[:, index]
    predictions["pairwise_prob_calm"] = pairwise_calm_probability
    predictions.to_csv(args.output_dir / "specialist_oof_predictions.csv", index=False)
    (args.output_dir / "specialist_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
