from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


LABEL_MAP = {
    "neu": "calm",
    "neutral": "calm",
    "hap": "joy",
    "happy": "joy",
    "ang": "anger",
    "angry": "anger",
    "sad": "sadness",
}
CLASSES = ["anger", "calm", "joy", "sadness"]


def _train_only_pilot(
    data: pd.DataFrame,
    max_per_class: int,
    seed: int,
) -> pd.DataFrame:
    train = data.loc[data["dataset_split"].astype(str).str.lower() == "train"].copy()
    required = {"file_path", "emotion", "speaker_id"}
    missing = sorted(required - set(train.columns))
    if missing:
        raise ValueError(f"В CSV отсутствуют столбцы: {', '.join(missing)}")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    _, validation_idx = next(
        splitter.split(train["file_path"], train["emotion"], train["speaker_id"])
    )
    validation = train.iloc[validation_idx].copy()
    parts = []
    for class_index, label in enumerate(CLASSES):
        rows = validation.loc[validation["emotion"] == label]
        parts.append(rows.sample(n=min(len(rows), max_per_class), random_state=seed + class_index))
    pilot = pd.concat(parts, ignore_index=True)
    return pilot.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _map_model_labels(model) -> dict[int, str]:
    mapped: dict[int, str] = {}
    for raw_index, raw_label in model.config.id2label.items():
        normalized = str(raw_label).strip().lower()
        if normalized not in LABEL_MAP:
            raise ValueError(f"Неизвестная teacher-метка: {raw_label}")
        mapped[int(raw_index)] = LABEL_MAP[normalized]
    if set(mapped.values()) != set(CLASSES):
        raise ValueError(f"Teacher не покрывает четыре класса: {mapped}")
    return mapped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train-only speaker-disjoint пилот внешнего audio teacher"
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="superb/wav2vec2-base-superb-er")
    parser.add_argument("--max-per-class", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    data = pd.read_csv(args.csv)
    pilot = _train_only_pilot(data, args.max_per_class, args.seed)
    pilot_speakers = set(pilot["speaker_id"].astype(str))
    official_test_speakers = set(
        data.loc[
            data["dataset_split"].astype(str).str.lower() == "test", "speaker_id"
        ].astype(str)
    )
    if pilot_speakers & official_test_speakers:
        raise RuntimeError("Pilot пересекается с official test по дикторам.")

    print(
        f"Pilot: {len(pilot)} записей, {len(pilot_speakers)} дикторов, "
        f"классы={pilot['emotion'].value_counts().to_dict()}",
        flush=True,
    )
    processor = AutoFeatureExtractor.from_pretrained(args.model)
    model = AutoModelForAudioClassification.from_pretrained(args.model)
    model.eval()
    id_to_class = _map_model_labels(model)

    output_rows = []
    started = time.perf_counter()
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(pilot), batch_size):
        batch = pilot.iloc[start : start + batch_size]
        waves = []
        valid_rows = []
        for _, row in batch.iterrows():
            try:
                wave, _ = librosa.load(str(row["file_path"]), sr=16000, mono=True)
                waves.append(np.asarray(wave, dtype=np.float32))
                valid_rows.append(row)
            except Exception as exc:
                output_rows.append(
                    {
                        "file_path": str(row["file_path"]),
                        "true_class": str(row["emotion"]),
                        "speaker_id": str(row["speaker_id"]),
                        "error": str(exc),
                    }
                )
        if waves:
            inputs = processor(
                waves,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            with torch.inference_mode():
                logits = model(**inputs).logits
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            for row, probs in zip(valid_rows, probabilities):
                prediction_index = int(np.argmax(probs))
                probability_map = {
                    id_to_class[index]: float(probs[index])
                    for index in range(len(probs))
                }
                sorted_values = sorted(probability_map.values(), reverse=True)
                entropy = -sum(
                    value * np.log(max(value, 1e-12)) for value in probability_map.values()
                ) / np.log(len(CLASSES))
                output_rows.append(
                    {
                        "file_path": str(row["file_path"]),
                        "true_class": str(row["emotion"]),
                        "speaker_id": str(row["speaker_id"]),
                        "teacher_class": id_to_class[prediction_index],
                        "teacher_confidence": float(sorted_values[0]),
                        "teacher_margin": float(sorted_values[0] - sorted_values[1]),
                        "teacher_entropy": float(entropy),
                        **{
                            f"prob_{label}": probability_map[label]
                            for label in CLASSES
                        },
                        "error": "",
                    }
                )
        done = min(start + batch_size, len(pilot))
        if done % 40 == 0 or done == len(pilot):
            elapsed = time.perf_counter() - started
            print(f"Teacher: {done}/{len(pilot)}, {elapsed:.1f} sec", flush=True)

    results = pd.DataFrame(output_rows)
    valid = results.loc[results.get("error", "") == ""].copy()
    y_true = valid["true_class"].astype(str)
    y_pred = valid["teacher_class"].astype(str)
    metrics = {
        "model": args.model,
        "protocol": "train_only_speaker_disjoint_pilot_official_test_untouched",
        "seed": args.seed,
        "records": int(len(valid)),
        "speakers": int(valid["speaker_id"].nunique()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0
        ),
        "mean_confidence": float(valid["teacher_confidence"].mean()),
        "mean_margin": float(valid["teacher_margin"].mean()),
        "mean_entropy": float(valid["teacher_entropy"].mean()),
        "duration_sec": float(time.perf_counter() - started),
        "official_test_used": False,
        "errors": int(len(results) - len(valid)),
    }
    csv_path = args.output_dir / "teacher_pilot_predictions.csv"
    json_path = args.output_dir / "teacher_pilot_metrics.json"
    results.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"Predictions: {csv_path}", flush=True)
    print(f"Metrics: {json_path}", flush=True)


if __name__ == "__main__":
    main()
