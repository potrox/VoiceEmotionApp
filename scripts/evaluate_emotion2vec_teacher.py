from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold


CLASSES = ["anger", "calm", "joy", "sadness"]
LABEL_MAP = {"angry": "anger", "neutral": "calm", "happy": "joy", "sad": "sadness"}


def _pilot(data: pd.DataFrame, max_per_class: int, seed: int) -> pd.DataFrame:
    train = data.loc[data["dataset_split"].astype(str).str.lower() == "train"].copy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    _, validation_idx = next(
        splitter.split(train["file_path"], train["emotion"], train["speaker_id"])
    )
    validation = train.iloc[validation_idx]
    parts = []
    for index, label in enumerate(CLASSES):
        rows = validation.loc[validation["emotion"] == label]
        parts.append(rows.sample(min(len(rows), max_per_class), random_state=seed + index))
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def _english_label(raw: str) -> str:
    return str(raw).split("/")[-1].strip().lower().replace("<unk>", "unknown")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-only emotion2vec+ teacher pilot")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-class", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.csv)
    pilot = _pilot(data, args.max_per_class, args.seed)
    test_speakers = set(
        data.loc[data["dataset_split"].astype(str).str.lower() == "test", "speaker_id"].astype(str)
    )
    if set(pilot["speaker_id"].astype(str)) & test_speakers:
        raise RuntimeError("Pilot пересекается с official test по дикторам.")

    logging.disable(logging.INFO)
    from funasr import AutoModel

    model_id = "emotion2vec/emotion2vec_plus_base"
    model = AutoModel(model=model_id, hub="hf", device="cpu", disable_update=True)

    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    batch_size = max(1, args.batch_size)
    for start in range(0, len(pilot), batch_size):
        batch = pilot.iloc[start : start + batch_size]
        generated = model.generate(
            input=batch["file_path"].astype(str).tolist(),
            granularity="utterance",
            extract_embedding=False,
        )
        by_key = {str(item["key"]): item for item in generated}
        for _, source in batch.iterrows():
            key = Path(str(source["file_path"])).stem
            item = by_key[key]
            labels = [_english_label(value) for value in item["labels"]]
            scores = np.asarray(item["scores"], dtype=float)
            native_index = int(np.argmax(scores))
            primary_scores = {
                LABEL_MAP[label]: float(score)
                for label, score in zip(labels, scores)
                if label in LABEL_MAP
            }
            forced_class = max(primary_scores, key=primary_scores.get)
            sorted_primary = sorted(primary_scores.values(), reverse=True)
            native_label = labels[native_index]
            rows.append(
                {
                    "file_path": str(source["file_path"]),
                    "speaker_id": str(source["speaker_id"]),
                    "true_class": str(source["emotion"]),
                    "native_label": native_label,
                    "native_confidence": float(scores[native_index]),
                    "teacher_class": LABEL_MAP.get(native_label, "abstain"),
                    "forced_primary_class": forced_class,
                    "forced_primary_confidence": float(primary_scores[forced_class]),
                    "primary_margin": float(sorted_primary[0] - sorted_primary[1]),
                    **{f"prob_{label}": primary_scores[label] for label in CLASSES},
                }
            )
        print(f"Teacher: {min(start + batch_size, len(pilot))}/{len(pilot)}", flush=True)

    result = pd.DataFrame(rows)
    covered = result.loc[result["teacher_class"] != "abstain"]
    y_true = result["true_class"].astype(str)
    y_forced = result["forced_primary_class"].astype(str)
    selective: dict[str, object] = {}
    for threshold in [0.4, 0.5, 0.6, 0.7, 0.8]:
        accepted = result.loc[
            (result["teacher_class"] != "abstain")
            & (result["native_confidence"] >= threshold)
        ]
        selective[str(threshold)] = {
            "records": int(len(accepted)),
            "coverage": float(len(accepted) / max(len(result), 1)),
            "accuracy": (
                float(accuracy_score(accepted["true_class"], accepted["teacher_class"]))
                if len(accepted)
                else None
            ),
        }
    metrics = {
        "model": model_id,
        "protocol": "official_train_only_speaker_disjoint_pilot_official_test_untouched",
        "records": int(len(result)),
        "speakers": int(result["speaker_id"].nunique()),
        "forced_primary_accuracy": float(accuracy_score(y_true, y_forced)),
        "forced_primary_macro_f1": float(f1_score(y_true, y_forced, average="macro")),
        "forced_primary_report": classification_report(
            y_true, y_forced, labels=CLASSES, output_dict=True, zero_division=0
        ),
        "native_primary_coverage": float(len(covered) / max(len(result), 1)),
        "native_primary_accuracy_when_covered": (
            float(accuracy_score(covered["true_class"], covered["teacher_class"]))
            if len(covered)
            else None
        ),
        "native_label_distribution": result["native_label"].value_counts().to_dict(),
        "selective_accuracy": selective,
        "duration_sec": float(time.perf_counter() - started),
        "official_test_used": False,
    }
    result.to_csv(args.output_dir / "emotion2vec_predictions.csv", index=False)
    (args.output_dir / "emotion2vec_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
