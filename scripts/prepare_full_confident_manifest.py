from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from prepare_dusha_archive import EMOTION_MAP, _read_raw


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a manifest containing every confident Dusha recording"
    )
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--split", choices=("train", "test", "all"), default="train"
    )
    args = parser.parse_args()

    raw = pd.concat(
        [_read_raw(args.metadata_dir, "train"), _read_raw(args.metadata_dir, "test")],
        ignore_index=True,
    )
    cache = args.metadata_dir / (
        f"dawid_skene_{args.threshold:.3f}_{int(args.iterations)}.parquet"
    )
    if not cache.is_file():
        raise FileNotFoundError(
            f"Expected the verified Dawid-Skene cache from the prior run: {cache}"
        )
    aggregated = pd.read_parquet(cache)
    metadata = raw.drop_duplicates("hash_id").copy()
    metadata = metadata[metadata["golden_emo"].isna()]
    metadata = metadata[metadata["source_id"].notna()]
    data = metadata.merge(aggregated, on="hash_id", how="inner")
    data = data[data["raw_emotion"].isin(EMOTION_MAP)].copy()
    if args.split != "all":
        data = data[data["dataset_split"].astype(str).eq(args.split)].copy()
    data["emotion"] = data["raw_emotion"].map(EMOTION_MAP)
    data["speaker_id"] = data["source_id"].astype(str)
    data["archive_member"] = data.apply(
        lambda row: (
            f"crowd_{row['dataset_split']}/"
            f"{str(row['audio_path']).replace(chr(92), '/')}"
        ),
        axis=1,
    )
    output_root = args.output_root.resolve()
    data["file_path"] = data["archive_member"].map(
        lambda member: str((output_root / Path(member)).resolve())
    )
    data["source"] = "Dusha crowd"
    data["language"] = "ru"
    columns = [
        "file_path",
        "emotion",
        "speaker_id",
        "dataset_split",
        "source",
        "language",
        "duration",
        "speaker_text",
        "annotation_confidence",
        "hash_id",
        "archive_member",
    ]
    data = data[columns].sort_values(["dataset_split", "emotion", "hash_id"])
    data = data.reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_csv, index=False, encoding="utf-8")
    summary = {
        "protocol": "all_dusha_records_from_verified_dawid_skene_cache",
        "split": args.split,
        "threshold": float(args.threshold),
        "records": int(len(data)),
        "speakers": int(data["speaker_id"].nunique()),
        "class_distribution": {
            str(key): int(value) for key, value in data["emotion"].value_counts().items()
        },
        "parameters_changed_from_v10": False,
    }
    args.output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
