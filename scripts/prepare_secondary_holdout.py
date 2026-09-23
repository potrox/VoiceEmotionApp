from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_dusha_archive import EMOTION_MAP, _read_raw, extract_selection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a recording-disjoint secondary holdout from unused official-test items"
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--previous-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=280)
    parser.add_argument("--max-per-speaker-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    raw = pd.concat(
        [_read_raw(args.metadata_dir, "train"), _read_raw(args.metadata_dir, "test")],
        ignore_index=True,
    )
    aggregation_path = args.metadata_dir / "dawid_skene_0.900_100.parquet"
    if not aggregation_path.is_file():
        raise FileNotFoundError(aggregation_path)
    aggregated = pd.read_parquet(aggregation_path)
    metadata = raw.drop_duplicates("hash_id").copy()
    metadata = metadata[metadata["golden_emo"].isna()]
    metadata = metadata[metadata["source_id"].notna()]
    data = metadata.merge(aggregated, on="hash_id", how="inner")
    data = data[
        data["dataset_split"].eq("test") & data["raw_emotion"].isin(EMOTION_MAP)
    ].copy()
    data["emotion"] = data["raw_emotion"].map(EMOTION_MAP)
    data["speaker_id"] = data["source_id"].astype(str)

    previous = pd.read_csv(args.previous_csv)
    used_hashes = set(previous["hash_id"].astype(str))
    data = data[~data["hash_id"].astype(str).isin(used_hashes)].copy()
    capped = pd.concat(
        [
            frame.sample(
                n=min(len(frame), int(args.max_per_speaker_class)),
                random_state=int(args.seed),
            )
            for _, frame in data.groupby(["emotion", "speaker_id"], sort=True)
        ],
        ignore_index=True,
    )
    available = capped["emotion"].value_counts()
    target = min(int(args.per_class), int(available.min()))
    selection = pd.concat(
        [
            frame.sample(n=target, random_state=int(args.seed))
            for _, frame in capped.groupby("emotion", sort=True)
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=int(args.seed)).reset_index(drop=True)

    selection["archive_member"] = selection.apply(
        lambda row: f"crowd_{row['dataset_split']}/{str(row['audio_path']).replace(chr(92), '/')}",
        axis=1,
    )
    selection["file_path"] = selection["archive_member"].map(
        lambda member: str((args.output_root / Path(member)).resolve())
    )
    selection["source"] = "Dusha crowd secondary holdout"
    selection["language"] = "ru"
    selection["dataset_split"] = "secondary_test"
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
    selection = selection[columns]
    old_test = previous[previous["dataset_split"].astype(str).str.lower().eq("test")]
    summary = {
        "protocol": "unused_official_test_recordings_same_test_speaker_pool",
        "seed": int(args.seed),
        "records": int(len(selection)),
        "records_per_class": int(target),
        "speakers": int(selection["speaker_id"].nunique()),
        "recording_overlap_with_previous_15k": 0,
        "speaker_overlap_with_previous_test": int(
            len(set(selection["speaker_id"]) & set(old_test["speaker_id"].astype(str)))
        ),
        "limitation": "Independent recordings, but not an independent speaker cohort from the previous official test.",
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(args.output_csv, index=False, encoding="utf-8")
    args.output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not args.skip_extract:
        extract_selection(args.archive, args.output_root, selection)


if __name__ == "__main__":
    main()
