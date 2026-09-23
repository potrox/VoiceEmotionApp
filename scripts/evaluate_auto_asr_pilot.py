"""Diagnostic ASR+emotion check; never use this reused test to tune the model."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.asr import LocalTranscriber


def _cache_path(cache_dir: Path, file_path: str) -> Path:
    path = Path(file_path)
    stat = path.stat()
    key = f"emotion2vec/emotion2vec_plus_base|{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return cache_dir / (hashlib.sha256(key.encode("utf-8")).hexdigest() + ".npy")


def _words(text: str) -> list[str]:
    return re.findall(r"[\w]+", str(text).lower(), flags=re.UNICODE)


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, token in enumerate(left, 1):
        current = [i]
        for j, target in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (token != target)))
        previous = current
    return previous[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--asr-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=20)
    args = parser.parse_args()

    data = pd.read_csv(args.manifest)
    data = data[data["dataset_split"].astype(str).eq("test")]
    sample = pd.concat(
        [part.sample(n=min(len(part), args.per_class), random_state=42) for _, part in data.groupby("emotion")],
        ignore_index=True,
    ).sample(frac=1, random_state=42).reset_index(drop=True)
    bundle = joblib.load(args.bundle)
    labels = bundle["label_encoder"].classes_
    audio_model = bundle["models"][bundle["best_model_name"]]
    text_model = bundle["text_model"]
    weight = float(bundle["fusion_audio_weight"])
    asr = LocalTranscriber(args.asr_cache)
    truth, audio_pred, manual_pred, auto_pred = [], [], [], []
    errors, words, empty, seconds = 0, 0, 0, []

    for index, row in sample.iterrows():
        embedding = np.load(_cache_path(args.embedding_cache, row["file_path"]), allow_pickle=False).reshape(1, -1)
        audio_probs = audio_model.predict_proba(embedding)[0]
        manual_probs = weight * audio_probs + (1.0 - weight) * text_model.predict_proba([str(row["speaker_text"])])[0]
        started = time.perf_counter()
        transcription = asr.transcribe_file(row["file_path"]).text
        seconds.append(time.perf_counter() - started)
        if transcription:
            auto_probs = weight * audio_probs + (1.0 - weight) * text_model.predict_proba([transcription])[0]
        else:
            empty += 1
            auto_probs = audio_probs
        ref_tokens = _words(row["speaker_text"])
        hyp_tokens = _words(transcription)
        errors += _edit_distance(ref_tokens, hyp_tokens)
        words += len(ref_tokens)
        truth.append(str(row["emotion"]))
        audio_pred.append(str(labels[np.argmax(audio_probs)]))
        manual_pred.append(str(labels[np.argmax(manual_probs)]))
        auto_pred.append(str(labels[np.argmax(auto_probs)]))
        if (index + 1) % 10 == 0:
            print(f"Processed {index + 1}/{len(sample)}; ASR seconds last={seconds[-1]:.2f}", flush=True)

    result = {
        "protocol": "reused_official_test_diagnostic_only_no_tuning",
        "records": len(sample),
        "per_class": sample["emotion"].value_counts().to_dict(),
        "model": "faster-whisper small CPU int8, ru, beam 3",
        "manual_text": {
            "accuracy": float(accuracy_score(truth, manual_pred)),
            "macro_f1": float(f1_score(truth, manual_pred, labels=labels, average="macro")),
            "per_class_f1": dict(zip(labels.tolist(), [float(x) for x in f1_score(truth, manual_pred, labels=labels, average=None)])),
        },
        "auto_asr": {
            "accuracy": float(accuracy_score(truth, auto_pred)),
            "macro_f1": float(f1_score(truth, auto_pred, labels=labels, average="macro")),
            "per_class_f1": dict(zip(labels.tolist(), [float(x) for x in f1_score(truth, auto_pred, labels=labels, average=None)])),
            "empty_transcripts": empty,
            "word_error_rate": errors / words if words else None,
            "mean_asr_seconds": float(np.mean(seconds)),
            "median_asr_seconds": float(np.median(seconds)),
        },
        "audio_only": {
            "accuracy": float(accuracy_score(truth, audio_pred)),
            "macro_f1": float(f1_score(truth, audio_pred, labels=labels, average="macro")),
            "per_class_f1": dict(zip(labels.tolist(), [float(x) for x in f1_score(truth, audio_pred, labels=labels, average=None)])),
        },
        "limitation": "Both models and the test have been inspected before. Small diagnostic slice, not a new independent estimate.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
