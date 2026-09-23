from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import time
from pathlib import Path

import pandas as pd


EMOTION_MAP = {
    "angry": "anger",
    "sad": "sadness",
    "neutral": "calm",
    "positive": "joy",
}


def _read_raw(metadata_dir: Path, split: str) -> pd.DataFrame:
    path = metadata_dir / f"crowd_{split}" / f"raw_crowd_{split}.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"Не найден файл метаданных: {path}")
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    required = {
        "hash_id", "audio_path", "duration", "annotator_emo", "annotator_id",
        "golden_emo", "speaker_text", "source_id",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"В {path.name} отсутствуют столбцы: {', '.join(missing)}")
    frame["dataset_split"] = split
    return frame


def aggregate_labels(raw: pd.DataFrame, threshold: float, iterations: int) -> pd.DataFrame:
    try:
        from crowdkit.aggregation import DawidSkene
    except ImportError as exc:
        raise RuntimeError(
            "Для официальной агрегации установите crowd-kit в отдельное окружение: pip install crowd-kit"
        ) from exc

    annotations = raw[["hash_id", "annotator_id", "annotator_emo"]].rename(
        columns={"hash_id": "task", "annotator_id": "worker", "annotator_emo": "label"}
    )
    probabilities = DawidSkene(n_iter=int(iterations)).fit_predict_proba(annotations)
    confidence = probabilities.max(axis=1)
    emotion = probabilities.idxmax(axis=1)
    aggregated = pd.DataFrame(
        {
            "hash_id": probabilities.index.astype(str),
            "raw_emotion": emotion.astype(str).to_numpy(),
            "annotation_confidence": confidence.astype(float).to_numpy(),
        }
    )
    return aggregated[aggregated["annotation_confidence"] >= float(threshold)].copy()


def _sample_split(
    frame: pd.DataFrame,
    split: str,
    per_class: int,
    max_per_speaker_class: int,
    seed: int,
) -> pd.DataFrame:
    split_frame = frame[frame["dataset_split"] == split].copy()
    capped_parts = [
        group.sample(n=min(len(group), int(max_per_speaker_class)), random_state=int(seed))
        for _, group in split_frame.groupby(["emotion", "speaker_id"], sort=True)
    ]
    if not capped_parts:
        raise ValueError(f"В {split} после фильтрации не осталось записей.")
    capped = pd.concat(capped_parts, ignore_index=True)
    available = capped["emotion"].value_counts()
    missing = sorted(set(EMOTION_MAP.values()) - set(available.index))
    if missing:
        raise ValueError(f"В {split} после фильтрации отсутствуют классы: {', '.join(missing)}")
    target = min(int(per_class), int(available.min()))
    sampled = pd.concat(
        [
            group.sample(n=target, random_state=int(seed))
            for _, group in capped.groupby("emotion", sort=True)
        ],
        ignore_index=True,
    )
    return sampled.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)


def build_selection(
    metadata_dir: Path,
    output_root: Path,
    threshold: float,
    iterations: int,
    train_per_class: int,
    test_per_class: int,
    train_max_per_speaker_class: int,
    test_max_per_speaker_class: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    raw_train = _read_raw(metadata_dir, "train")
    raw_test = _read_raw(metadata_dir, "test")
    raw = pd.concat([raw_train, raw_test], ignore_index=True)
    started = time.perf_counter()
    cache_path = metadata_dir / f"dawid_skene_{threshold:.3f}_{int(iterations)}.parquet"
    if cache_path.is_file():
        aggregated = pd.read_parquet(cache_path)
    else:
        aggregated = aggregate_labels(raw, threshold=threshold, iterations=iterations)
        aggregated.to_parquet(cache_path, index=False)

    metadata = raw.drop_duplicates("hash_id").copy()
    metadata = metadata[metadata["golden_emo"].isna()]
    metadata = metadata[metadata["source_id"].notna()]
    data = metadata.merge(aggregated, on="hash_id", how="inner")
    data = data[data["raw_emotion"].isin(EMOTION_MAP)].copy()
    data["emotion"] = data["raw_emotion"].map(EMOTION_MAP)
    data["speaker_id"] = data["source_id"].astype(str)
    data["archive_member"] = data.apply(
        lambda row: f"crowd_{row['dataset_split']}/{str(row['audio_path']).replace(chr(92), '/')}",
        axis=1,
    )
    data["file_path"] = data["archive_member"].map(
        lambda member: str((output_root / Path(member)).resolve())
    )

    train = _sample_split(
        data, "train", train_per_class, train_max_per_speaker_class, seed
    )
    test = _sample_split(
        data, "test", test_per_class, test_max_per_speaker_class, seed
    )
    selection = pd.concat([train, test], ignore_index=True)
    train_speakers = set(train["speaker_id"])
    test_speakers = set(test["speaker_id"])
    overlap = sorted(train_speakers & test_speakers)
    if overlap:
        raise RuntimeError(f"Официальный split содержит пересечение дикторов: {overlap[:10]}")

    columns = [
        "file_path", "emotion", "speaker_id", "dataset_split", "source", "language",
        "duration", "speaker_text", "annotation_confidence", "hash_id", "archive_member",
    ]
    selection["source"] = "Dusha crowd"
    selection["language"] = "ru"
    selection = selection[columns]
    summary = {
        "protocol": "official_dusha_split_dawid_skene_0.9",
        "seed": int(seed),
        "dawid_skene_threshold": float(threshold),
        "dawid_skene_iterations": int(iterations),
        "aggregation_seconds": round(time.perf_counter() - started, 3),
        "raw_annotation_rows": int(len(raw)),
        "raw_unique_recordings": int(raw["hash_id"].nunique()),
        "aggregated_confident_recordings": int(len(data)),
        "selected_recordings": int(len(selection)),
        "class_distribution": selection.groupby(["dataset_split", "emotion"]).size().to_dict(),
        "train_speakers": int(train["speaker_id"].nunique()),
        "test_speakers": int(test["speaker_id"].nunique()),
        "speaker_overlap": 0,
    }
    summary["class_distribution"] = {
        f"{split}:{emotion}": int(count)
        for (split, emotion), count in summary["class_distribution"].items()
    }
    return selection, summary


def extract_selection(archive: Path, output_root: Path, selection: pd.DataFrame) -> None:
    wanted = set(selection["archive_member"].astype(str))
    output_root = output_root.resolve()
    extracted = 0
    with tarfile.open(archive, mode="r:") as stream:
        for member in stream:
            if member.name not in wanted:
                continue
            destination = (output_root / Path(member.name)).resolve()
            if output_root not in destination.parents:
                raise RuntimeError(f"Небезопасный путь в архиве: {member.name}")
            source = stream.extractfile(member)
            if source is None:
                raise RuntimeError(f"Не удалось прочитать {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            extracted += 1
            if extracted % 500 == 0:
                print(f"Распаковано {extracted} из {len(wanted)}", flush=True)
    missing = wanted - {
        str(path.relative_to(output_root)).replace("\\", "/")
        for path in output_root.rglob("*.wav")
    }
    if missing:
        raise RuntimeError(f"В архиве не найдено файлов: {len(missing)}; пример: {sorted(missing)[:3]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Подготовить честную подвыборку Dusha прямо из crowd.tar")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=3000)
    parser.add_argument("--test-per-class", type=int, default=750)
    parser.add_argument("--train-max-per-speaker-class", type=int, default=12)
    parser.add_argument("--test-max-per-speaker-class", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()

    if args.extract_only:
        if not args.output_csv.is_file():
            raise FileNotFoundError(f"Не найден готовый CSV: {args.output_csv}")
        selection = pd.read_csv(args.output_csv)
        extract_selection(args.archive, args.output_root, selection)
        print(f"Распаковка завершена: {args.output_root}", flush=True)
        return

    selection, summary = build_selection(
        metadata_dir=args.metadata_dir,
        output_root=args.output_root,
        threshold=args.threshold,
        iterations=args.iterations,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
        train_max_per_speaker_class=args.train_max_per_speaker_class,
        test_max_per_speaker_class=args.test_max_per_speaker_class,
        seed=args.seed,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(args.output_csv, index=False, encoding="utf-8")
    summary_path = args.output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not args.skip_extract:
        extract_selection(args.archive, args.output_root, selection)
    print(f"CSV: {args.output_csv}", flush=True)
    print(f"Данные: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
