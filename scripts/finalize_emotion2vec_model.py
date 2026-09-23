from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, label_binarize

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.honest_models import ClassBiasClassifier, _tune_class_biases
from scripts.multimodal_components import MODEL_ID, _cache_path
from scripts.run_emotion2vec_embedding_pilot import _fit_embedding_baseline
from scripts.experiment_common import CLASSES, sample_train as _sample_train


def _load_embeddings(data: pd.DataFrame, cache_dir: Path) -> np.ndarray:
    paths = [_cache_path(cache_dir, str(value)) for value in data["file_path"]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} cached embeddings; first: {missing[0]}"
        )
    matrix = np.vstack([np.load(path, allow_pickle=False) for path in paths])
    if matrix.shape != (len(data), 768):
        raise ValueError(f"Unexpected embedding matrix shape: {matrix.shape}")
    return matrix.astype(np.float32, copy=False)


def _probability_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    encoder: LabelEncoder,
) -> dict[str, Any]:
    predictions = np.argmax(probabilities, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=np.arange(len(encoder.classes_)),
        zero_division=0,
    )
    binary = label_binarize(y_true, classes=np.arange(len(encoder.classes_)))
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted")),
        "roc_auc_macro_ovr": float(
            roc_auc_score(binary, probabilities, average="macro", multi_class="ovr")
        ),
        "average_precision_macro": float(
            average_precision_score(binary, probabilities, average="macro")
        ),
        "log_loss": float(log_loss(y_true, probabilities)),
        "per_class": {
            str(label): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
                "roc_auc": float(roc_auc_score(binary[:, index], probabilities[:, index])),
                "average_precision": float(
                    average_precision_score(binary[:, index], probabilities[:, index])
                ),
            }
            for index, label in enumerate(encoder.classes_)
        },
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            target_names=[str(value) for value in encoder.classes_],
            zero_division=0,
        ),
    }


def _speaker_bootstrap_interval(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    speakers: np.ndarray,
    seed: int,
    repeats: int = 1000,
) -> list[float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(speakers)
    by_speaker = {value: np.flatnonzero(speakers == value) for value in unique}
    scores: list[float] = []
    for _ in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_speaker[value] for value in sampled])
        if len(np.unique(y_true[indices])) != probabilities.shape[1]:
            continue
        pred = np.argmax(probabilities[indices], axis=1)
        scores.append(float(f1_score(y_true[indices], pred, average="macro")))
    if not scores:
        return []
    return [float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze a train-selected emotion2vec classifier and evaluate official test once"
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(args.csv)
    train = _sample_train(source, max_per_class=3000, seed=args.seed)
    test = source[source["dataset_split"].astype(str).str.lower().eq("test")].copy()
    test = test.sort_values(["emotion", "file_path"]).reset_index(drop=True)
    train_speakers = set(train["speaker_id"].astype(str))
    test_speakers = set(test["speaker_id"].astype(str))
    overlap = train_speakers & test_speakers
    if overlap:
        raise RuntimeError(f"Official train/test speaker overlap: {len(overlap)}")

    encoder = LabelEncoder().fit(CLASSES)
    y_train = encoder.transform(train["emotion"].astype(str))
    y_test = encoder.transform(test["emotion"].astype(str))
    groups_train = train["speaker_id"].astype(str).to_numpy()
    groups_test = test["speaker_id"].astype(str).to_numpy()
    X_train = _load_embeddings(train, args.embedding_cache_dir)
    X_test = _load_embeddings(test, args.embedding_cache_dir)

    oof = pd.read_csv(args.oof_predictions)
    probability_columns = [f"embedding_prob_{label}" for label in encoder.classes_]
    aligned = train[["file_path", "emotion"]].merge(
        oof[["file_path", *probability_columns]],
        on="file_path",
        how="left",
        validate="one_to_one",
    )
    if aligned[probability_columns].isna().any().any():
        raise RuntimeError("Could not align all train OOF predictions.")
    oof_probabilities = aligned[probability_columns].to_numpy(dtype=float)
    biases, oof_raw_f1, oof_biased_f1 = _tune_class_biases(y_train, oof_probabilities)

    started = time.perf_counter()
    model, selected_c, inner_scores = _fit_embedding_baseline(
        X_train,
        y_train,
        groups_train,
        args.seed,
        inner_splits=5,
    )
    raw_probabilities = model.predict_proba(X_test)
    deployed_model = ClassBiasClassifier(model, biases)
    biased_probabilities = deployed_model.predict_proba(X_test)
    raw_metrics = _probability_metrics(y_test, raw_probabilities, encoder)
    final_metrics = _probability_metrics(y_test, biased_probabilities, encoder)
    final_metrics["speaker_bootstrap_macro_f1_95ci"] = _speaker_bootstrap_interval(
        y_test, biased_probabilities, groups_test, args.seed
    )

    benchmark_repeats = 200
    benchmark_started = time.perf_counter()
    for _ in range(benchmark_repeats):
        deployed_model.predict_proba(X_test[:1])
    classifier_latency_ms = (
        (time.perf_counter() - benchmark_started) * 1000.0 / benchmark_repeats
    )

    report: dict[str, Any] = {
        "protocol": "frozen_train_selected_model_single_official_test_evaluation",
        "model": MODEL_ID,
        "train_records": int(len(train)),
        "test_records": int(len(test)),
        "train_speakers": int(len(train_speakers)),
        "test_speakers": int(len(test_speakers)),
        "speaker_overlap": 0,
        "selection": {
            "selected_logistic_c": float(selected_c),
            "inner_grouped_cv_macro_f1": inner_scores,
            "nested_grouped_oof_macro_f1": float(oof_raw_f1),
            "oof_bias_tuned_macro_f1": float(oof_biased_f1),
            "class_log_probability_biases": {
                str(label): float(biases[index])
                for index, label in enumerate(encoder.classes_)
            },
            "test_used_for_selection": False,
        },
        "official_test_raw": raw_metrics,
        "official_test_deployed": final_metrics,
        "oof_to_test_macro_f1_gap": float(oof_biased_f1 - final_metrics["macro_f1"]),
        "classifier_latency_ms_single_record": float(classifier_latency_ms),
        "training_and_evaluation_sec": float(time.perf_counter() - started),
    }

    model_name = "Emotion2Vec + Logistic Regression"
    bundle = {
        "models": {model_name: deployed_model},
        "best_model_name": model_name,
        "scaler": None,
        "label_encoder": encoder,
        "feature_names": [f"emotion2vec_{index:03d}" for index in range(768)],
        "selector": None,
        "preprocessing_params": {
            "preprocessing_in_model": True,
            "feature_backend": "emotion2vec",
            "emotion2vec_model": MODEL_ID,
            "embedding_dimension": 768,
            "reliability_threshold": 0.55,
            "reliability_threshold_source": "train grouped OOF only",
            "sample_rate": 16000,
            "denoise": False,
            "trim_silence": False,
            "normalize_amplitude": False,
            "evaluation_protocol": report["protocol"],
        },
        "metrics": {model_name: final_metrics},
        "dataset_info": {
            "name": "Dusha Crowd 15k balanced subset",
            "train_records": int(len(train)),
            "test_records": int(len(test)),
            "train_speakers": int(len(train_speakers)),
            "test_speakers": int(len(test_speakers)),
            "speaker_overlap": 0,
            "test_used_for_selection": False,
        },
        "ensemble_weights": {},
        "created_at": time.strftime("%Y-%m-%d_%H-%M-%S"),
    }
    joblib.dump(bundle, args.output_dir / "emotion2vec_final_bundle.pkl")
    predictions = test[["file_path", "emotion", "speaker_id"]].copy()
    predictions["predicted"] = encoder.inverse_transform(
        np.argmax(biased_probabilities, axis=1)
    )
    for index, label in enumerate(encoder.classes_):
        predictions[f"prob_{label}"] = biased_probabilities[:, index]
    predictions.to_csv(args.output_dir / "emotion2vec_official_test_predictions.csv", index=False)
    (args.output_dir / "emotion2vec_final_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
