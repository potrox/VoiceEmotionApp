from __future__ import annotations

import argparse
import contextlib
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.multimodal_components import MODEL_ID, _cache_path
from scripts.experiment_common import sample_train as _sample_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable sharded emotion2vec extraction")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--max-per-class", type=int, default=3000)
    parser.add_argument(
        "--dataset-split",
        choices=("train", "test", "all"),
        default="train",
        help="Extract only the requested official split (default: train).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--ncpu", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count).")
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(args.csv)
    if args.dataset_split == "train":
        data = _sample_train(source, args.max_per_class, args.seed)
    elif args.dataset_split == "test":
        data = source[
            source["dataset_split"].astype(str).str.lower().eq("test")
        ].copy()
        data = (
            data.groupby("emotion", group_keys=False)
            .apply(
                lambda frame: frame.sample(
                    n=min(len(frame), args.max_per_class), random_state=args.seed
                )
            )
            .reset_index(drop=True)
        )
    else:
        data = source.copy().reset_index(drop=True)
    if data.empty:
        raise ValueError(f"No records found for official split: {args.dataset_split}")
    candidate_indices = [
        index for index in range(len(data)) if index % args.shard_count == args.shard_index
    ]
    missing = [
        index
        for index in candidate_indices
        if not _cache_path(args.cache_dir, str(data.iloc[index]["file_path"])).is_file()
    ]
    print(
        f"Shard {args.shard_index + 1}/{args.shard_count}: "
        f"{len(candidate_indices)} assigned, {len(missing)} missing",
        flush=True,
    )
    if not missing:
        return

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
                ncpu=args.ncpu,
            )
    batch_size = max(1, args.batch_size)
    for start in range(0, len(missing), batch_size):
        indices = missing[start : start + batch_size]
        paths = data.iloc[indices]["file_path"].astype(str).tolist()
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                generated = model.generate(
                    input=paths,
                    granularity="utterance",
                    extract_embedding=True,
                    batch_size=batch_size,
                )
        by_key = {
            str(item["key"]): np.asarray(item["feats"], dtype=np.float32)
            for item in generated
        }
        for index in indices:
            file_path = str(data.iloc[index]["file_path"])
            key = Path(file_path).stem
            embedding = by_key[key].reshape(-1)
            if embedding.shape != (768,):
                raise ValueError(f"Unexpected embedding shape {embedding.shape} for {key}")
            np.save(_cache_path(args.cache_dir, file_path), embedding, allow_pickle=False)
        done = min(start + batch_size, len(missing))
        if done % 40 == 0 or done == len(missing):
            print(f"Shard {args.shard_index + 1}: {done}/{len(missing)}", flush=True)


if __name__ == "__main__":
    main()
