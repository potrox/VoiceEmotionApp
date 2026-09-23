"""Shared holdout metrics, speaker bootstrap and fusion evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, label_binarize

from scripts.multimodal_components import _load_embeddings


def _metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    encoder: LabelEncoder,
) -> dict[str, Any]:
    prediction = np.argmax(probabilities, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        prediction,
        labels=np.arange(len(encoder.classes_)),
        zero_division=0,
    )
    binary = label_binarize(y_true, classes=np.arange(len(encoder.classes_)))
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro")),
        "roc_auc_macro_ovr": float(
            roc_auc_score(binary, probabilities, average="macro", multi_class="ovr")
        ),
        "average_precision_macro": float(
            average_precision_score(binary, probabilities, average="macro")
        ),
        "per_class": {
            str(label): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(encoder.classes_)
        },
        "confusion_matrix": confusion_matrix(y_true, prediction).tolist(),
    }


def _speaker_bootstrap(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    seed: int,
    repeats: int = 1000,
) -> list[float]:
    rng = np.random.default_rng(seed)
    speakers = np.unique(groups)
    by_speaker = {speaker: np.flatnonzero(groups == speaker) for speaker in speakers}
    scores: list[float] = []
    for _ in range(repeats):
        sampled = rng.choice(speakers, size=len(speakers), replace=True)
        indices = np.concatenate([by_speaker[speaker] for speaker in sampled])
        if len(np.unique(y_true[indices])) < probabilities.shape[1]:
            continue
        scores.append(
            float(
                f1_score(
                    y_true[indices],
                    np.argmax(probabilities[indices], axis=1),
                    average="macro",
                )
            )
        )
    return [float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975))]


def _evaluate_manifest(
    name: str,
    data: pd.DataFrame,
    cache_dir: Path,
    audio_model: Any,
    text_model: Any,
    audio_weight: float,
    encoder: LabelEncoder,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    X = _load_embeddings(data, cache_dir)
    texts = data["speaker_text"].fillna("").astype(str).to_numpy()
    y = encoder.transform(data["emotion"].astype(str))
    audio_probabilities = audio_model.predict_proba(X)
    text_probabilities = text_model.predict_proba(texts)
    fusion_probabilities = (
        audio_weight * audio_probabilities
        + (1.0 - audio_weight) * text_probabilities
    )
    audio_metrics = _metrics(y, audio_probabilities, encoder)
    fusion_metrics = _metrics(y, fusion_probabilities, encoder)
    fusion_metrics["speaker_bootstrap_macro_f1_95ci"] = _speaker_bootstrap(
        y,
        fusion_probabilities,
        data["speaker_id"].astype(str).to_numpy(),
        seed,
    )
    report = {
        "name": name,
        "records": int(len(data)),
        "speakers": int(data["speaker_id"].astype(str).nunique()),
        "audio": audio_metrics,
        "fusion": fusion_metrics,
        "fusion_gain_macro_f1": float(
            fusion_metrics["macro_f1"] - audio_metrics["macro_f1"]
        ),
    }
    predictions = data[["file_path", "emotion", "speaker_id", "speaker_text"]].copy()
    predictions["audio_class"] = encoder.inverse_transform(
        np.argmax(audio_probabilities, axis=1)
    )
    predictions["fusion_class"] = encoder.inverse_transform(
        np.argmax(fusion_probabilities, axis=1)
    )
    for index, label in enumerate(encoder.classes_):
        predictions[f"fusion_prob_{label}"] = fusion_probabilities[:, index]
    return report, predictions
