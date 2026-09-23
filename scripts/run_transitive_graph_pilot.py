from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.multimodal_components import _load_embeddings, _audio_model
from scripts.run_emotion2vec_embedding_pilot import _fit_embedding_baseline
from scripts.experiment_common import CLASSES, classification_metrics as _metrics, sample_train as _sample_train


class TransitiveGraphClassifier:
    """kNN graph with soft two-hop label propagation on training nodes."""

    def __init__(self, neighbors: int, alpha: float, feature_count: int = 128):
        self.neighbors = int(neighbors)
        self.alpha = float(alpha)
        self.preprocessor = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(f_classif, k=feature_count)),
                ("normalizer", Normalizer(norm="l2")),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TransitiveGraphClassifier":
        values = self.preprocessor.fit_transform(X, y)
        count = len(values)
        k = min(self.neighbors + 1, count)
        self.index = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=-1)
        self.index.fit(values)
        distances, indices = self.index.kneighbors(values)
        distances, indices = distances[:, 1:], indices[:, 1:]
        weights = np.exp(-distances / 0.20)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        rows = np.repeat(np.arange(count), indices.shape[1])
        graph = sparse.csr_matrix(
            (weights.ravel(), (rows, indices.ravel())), shape=(count, count)
        )
        labels = np.eye(len(CLASSES), dtype=float)[y]
        propagated = labels.copy()
        # Two updates encode direct and A->B->C relations. The original labels are
        # retained as anchors so noisy neighborhoods cannot freely overwrite them.
        for _ in range(2):
            propagated = (1.0 - self.alpha) * labels + self.alpha * graph.dot(propagated)
        self.train_posteriors = propagated / np.maximum(
            propagated.sum(axis=1, keepdims=True), 1e-12
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        values = self.preprocessor.transform(X)
        k = min(self.neighbors, len(self.train_posteriors))
        distances, indices = self.index.kneighbors(values, n_neighbors=k)
        weights = np.exp(-distances / 0.20)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        probabilities = np.einsum(
            "ij,ijk->ik", weights, self.train_posteriors[indices]
        )
        return probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)


def _inner_select(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    logistic_c: float,
    seed: int,
) -> tuple[dict, dict[str, float]]:
    candidates = [
        (neighbors, alpha, blend)
        for neighbors in (7, 15, 31)
        for alpha in (0.25, 0.50, 0.75)
        for blend in (0.15, 0.30, 0.50)
    ]
    scores = {f"k={k},a={a},b={b}": [] for k, a, b in candidates}
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    for train_idx, validation_idx in splitter.split(X, y, groups):
        logistic = _audio_model(logistic_c, X.shape[1], seed)
        logistic.fit(X[train_idx], y[train_idx])
        logistic_probabilities = logistic.predict_proba(X[validation_idx])
        graph_cache: dict[tuple[int, float], np.ndarray] = {}
        for neighbors, alpha, blend in candidates:
            graph_key = (neighbors, alpha)
            if graph_key not in graph_cache:
                graph = TransitiveGraphClassifier(neighbors, alpha).fit(
                    X[train_idx], y[train_idx]
                )
                graph_cache[graph_key] = graph.predict_proba(X[validation_idx])
            probabilities = (
                (1.0 - blend) * logistic_probabilities
                + blend * graph_cache[graph_key]
            )
            key = f"k={neighbors},a={alpha},b={blend}"
            scores[key].append(
                float(
                    f1_score(
                        y[validation_idx],
                        np.argmax(probabilities, axis=1),
                        average="macro",
                    )
                )
            )
    means = {key: float(np.mean(values)) for key, values in scores.items()}
    best_key = max(means, key=means.get)
    fields = dict(part.split("=") for part in best_key.split(","))
    return {
        "neighbors": int(fields["k"]),
        "alpha": float(fields["a"]),
        "blend": float(fields["b"]),
    }, means


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested grouped transitive graph pilot")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-class", type=int, default=250)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = _sample_train(pd.read_csv(args.csv), args.max_per_class, args.seed)
    X = _load_embeddings(data, args.embedding_cache_dir)
    groups = data["speaker_id"].astype(str).to_numpy()
    encoder = LabelEncoder().fit(CLASSES)
    y = encoder.transform(data["emotion"].astype(str))
    baseline_probabilities = np.zeros((len(y), len(CLASSES)), dtype=float)
    graph_probabilities = np.zeros_like(baseline_probabilities)
    fold_details: list[dict] = []
    started = time.perf_counter()
    outer = StratifiedGroupKFold(
        n_splits=args.outer_splits, shuffle=True, random_state=args.seed
    )
    for fold, (train_idx, validation_idx) in enumerate(outer.split(X, y, groups)):
        logistic, selected_c, logistic_scores = _fit_embedding_baseline(
            X[train_idx], y[train_idx], groups[train_idx], args.seed + fold, 3
        )
        base = logistic.predict_proba(X[validation_idx])
        baseline_probabilities[validation_idx] = base
        selected, inner_scores = _inner_select(
            X[train_idx],
            y[train_idx],
            groups[train_idx],
            selected_c,
            args.seed + 100 + fold,
        )
        graph = TransitiveGraphClassifier(
            selected["neighbors"], selected["alpha"]
        ).fit(X[train_idx], y[train_idx])
        graph_only = graph.predict_proba(X[validation_idx])
        graph_probabilities[validation_idx] = (
            (1.0 - selected["blend"]) * base + selected["blend"] * graph_only
        )
        fold_details.append(
            {
                "fold": fold,
                "logistic_c": float(selected_c),
                "logistic_inner_scores": logistic_scores,
                "selected_graph": selected,
                "selected_graph_inner_macro_f1": float(max(inner_scores.values())),
            }
        )
        print(f"Fold {fold + 1}/{args.outer_splits} complete", flush=True)

    baseline_metrics = _metrics(y, np.argmax(baseline_probabilities, axis=1))
    graph_metrics = _metrics(y, np.argmax(graph_probabilities, axis=1))
    result = {
        "protocol": "official_train_only_nested_speaker_grouped_two_hop_graph_pilot",
        "records": int(len(y)),
        "speakers": int(len(np.unique(groups))),
        "official_test_used": False,
        "relation": (
            "Same-class train nodes are anchors; two graph updates add A-B and B-C paths "
            "without using validation labels."
        ),
        "metrics": {
            "logistic": baseline_metrics,
            "logistic_plus_transitive_graph": graph_metrics,
        },
        "gain_macro_f1": float(
            graph_metrics["macro_f1"] - baseline_metrics["macro_f1"]
        ),
        "fold_details": fold_details,
        "duration_sec": float(time.perf_counter() - started),
    }
    predictions = data[["file_path", "emotion", "speaker_id"]].copy()
    predictions["logistic_class"] = encoder.inverse_transform(
        np.argmax(baseline_probabilities, axis=1)
    )
    predictions["graph_class"] = encoder.inverse_transform(
        np.argmax(graph_probabilities, axis=1)
    )
    predictions.to_csv(args.output_dir / "transitive_graph_oof.csv", index=False)
    (args.output_dir / "transitive_graph_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
