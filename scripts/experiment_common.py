"""Shared dataset sampling and class-wise metrics for offline experiments.

These helpers have no dependency on a particular pilot or finalization script.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support


CLASSES = ["anger", "calm", "joy", "sadness"]


def sample_train(data: pd.DataFrame, max_per_class: int | None, seed: int) -> pd.DataFrame:
    train = data.loc[data["dataset_split"].astype(str).str.lower() == "train"].copy()
    if not max_per_class:
        return train.reset_index(drop=True)
    parts = []
    for index, label in enumerate(CLASSES):
        rows = train.loc[train["emotion"] == label]
        parts.append(rows.sample(min(max_per_class, len(rows)), random_state=seed + index))
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def classification_metrics(
    y: np.ndarray, predicted: np.ndarray, accepted: np.ndarray | None = None
) -> dict[str, Any]:
    precision, recall, class_f1, support = precision_recall_fscore_support(
        y, predicted, labels=np.arange(len(CLASSES)), zero_division=0
    )
    result: dict[str, Any] = {
        "macro_f1": float(f1_score(y, predicted, average="macro")),
        "per_class": {
            CLASSES[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
                "support": int(support[index]),
            }
            for index in range(len(CLASSES))
        },
    }
    if accepted is not None:
        changed = accepted & (predicted >= 0)
        result["accepted_records"] = int(np.sum(changed))
        result["coverage"] = float(np.mean(changed))
        result["accepted_accuracy"] = (
            float(np.mean(predicted[changed] == y[changed])) if np.any(changed) else None
        )
    return result
