"""Reusable audio/text model construction and group-safe selection.

No dependency on command-line experiment entry points.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_ID = "emotion2vec/emotion2vec_plus_base"


def _cache_path(cache_dir: Path, file_path: str) -> Path:
    path = Path(file_path)
    stat = path.stat()
    payload = f"{MODEL_ID}|{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return cache_dir / f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()}.npy"


def _embedding_pipeline(model: Any, feature_count: int, scaled: bool = True) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scaled:
        steps.append(("scaler", StandardScaler()))
    steps.extend(
        [
            ("selector", SelectKBest(f_classif, k=min(256, feature_count))),
            ("model", model),
        ]
    )
    return Pipeline(steps)


C_VALUES = (0.01, 0.03, 0.1, 0.3)


def _load_embeddings(data: pd.DataFrame, cache_dir: Path) -> np.ndarray:
    paths = [_cache_path(cache_dir, str(value)) for value in data["file_path"]]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} embeddings; first: {missing[0]}")
    return np.vstack([np.load(path, allow_pickle=False) for path in paths]).astype(np.float32)


def _splits(X: Any, y: np.ndarray, groups: np.ndarray, seed: int, n: int = 3):
    return list(
        StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed).split(
            X, y, groups
        )
    )


def _audio_model(c_value: float, feature_count: int, seed: int) -> Pipeline:
    return _embedding_pipeline(
        LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            max_iter=1500,
            solver="lbfgs",
            random_state=seed,
        ),
        feature_count,
    )


def _text_model(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=20000,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    C=float(c_value),
                    class_weight="balanced",
                    max_iter=1500,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def _select_model_and_inner_probabilities(
    X: Any,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    family: str,
) -> tuple[Any, np.ndarray, float, dict[str, float]]:
    splits = _splits(X, y, groups, seed)
    scores: dict[str, float] = {}
    models: dict[float, Any] = {}
    for c_value in C_VALUES:
        model = (
            _audio_model(c_value, X.shape[1], seed)
            if family == "audio"
            else _text_model(c_value, seed)
        )
        models[c_value] = model
        scores[str(c_value)] = float(
            np.mean(
                cross_val_score(
                    model,
                    X,
                    y,
                    cv=splits,
                    scoring="f1_macro",
                    n_jobs=1,
                )
            )
        )
    best = max(scores.values())
    selected_c = min(
        float(key) for key, value in scores.items() if value >= best - 0.005
    )
    model = models[selected_c]
    inner_probabilities = cross_val_predict(
        clone(model),
        X,
        y,
        cv=splits,
        method="predict_proba",
        n_jobs=1,
    )
    model.fit(X, y)
    return model, inner_probabilities, selected_c, scores
