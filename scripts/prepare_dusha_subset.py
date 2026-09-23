from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


PATH_COLUMNS = ("file_path", "audio_path", "audio_filepath", "path", "wav_path", "wav", "audio")
EMOTION_COLUMNS = ("emotion", "emotion_label", "label", "class", "target", "annotator_emo")
SPEAKER_COLUMNS = ("speaker_id", "speaker", "client_id", "user_id", "actor", "source_id")

EMOTION_MAP = {
    "positive": "joy",
    "happiness": "joy",
    "happy": "joy",
    "joy": "joy",
    "sad": "sadness",
    "sadness": "sadness",
    "anger": "anger",
    "angry": "anger",
    "neutral": "calm",
    "calm": "calm",
}


def _find_column(columns: Iterable[str], candidates: Iterable[str], label: str) -> str:
    lookup = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    raise ValueError(f"Не найден столбец {label}. Доступно: {', '.join(map(str, columns))}")


def _read_manifest(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        rows = []
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _resolve_audio_path(value: object, audio_root: Path) -> str:
    path = Path(str(value).strip())
    if not path.is_absolute():
        path = audio_root / path
    return str(path.resolve())


def build_subset(
    manifest: Path,
    audio_root: Path,
    output: Path,
    max_per_class: int,
    max_per_speaker_class: int,
    seed: int,
) -> pd.DataFrame:
    source = _read_manifest(manifest)
    path_col = _find_column(source.columns, PATH_COLUMNS, "с путём к аудио")
    emotion_col = _find_column(source.columns, EMOTION_COLUMNS, "с эмоцией")
    speaker_col = _find_column(source.columns, SPEAKER_COLUMNS, "speaker_id")

    data = pd.DataFrame(
        {
            "file_path": source[path_col].map(lambda value: _resolve_audio_path(value, audio_root)),
            "emotion": source[emotion_col].astype(str).str.strip().str.lower().map(EMOTION_MAP),
            "speaker_id": source[speaker_col].astype(str).str.strip(),
        }
    )
    data = data.dropna(subset=["emotion"])
    data = data[~data["speaker_id"].str.lower().isin({"", "unknown", "none", "nan"})]
    data = data[data["file_path"].map(lambda value: Path(value).is_file())]
    if data.empty:
        raise ValueError(
            "После проверки manifest не осталось подходящих аудиофайлов. "
            "Проверьте --audio-root, названия столбцов и пути в manifest."
        )

    capped = pd.concat(
        [
            group.sample(
                n=min(len(group), int(max_per_speaker_class)),
                random_state=int(seed),
            )
            for _, group in data.groupby(["emotion", "speaker_id"])
        ],
        ignore_index=True,
    )
    balanced_parts = []
    for emotion, group in capped.groupby("emotion"):
        balanced_parts.append(
            group.sample(n=min(len(group), int(max_per_class)), random_state=int(seed))
        )
    if not balanced_parts:
        raise ValueError("Не удалось сформировать ни одного класса для подвыборки.")
    result = pd.concat(balanced_parts, ignore_index=True)
    result = result.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
    result["source"] = "Dusha crowd"
    result["language"] = "ru"

    speaker_counts = result.groupby("emotion")["speaker_id"].nunique()
    if len(speaker_counts) != 4 or int(speaker_counts.min()) < 3:
        raise ValueError(
            "После фильтрации в каждом классе должно остаться минимум три диктора. "
            f"Получено: {speaker_counts.to_dict()}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Создать воспроизводимую сбалансированную подвыборку Dusha для VoiceEmotionApp."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL/TSV/CSV manifest Dusha")
    parser.add_argument("--audio-root", type=Path, required=True, help="Корень распакованных аудиофайлов")
    parser.add_argument("--output", type=Path, required=True, help="Куда сохранить CSV приложения")
    parser.add_argument("--max-per-class", type=int, default=3000)
    parser.add_argument("--max-per-speaker-class", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = build_subset(
        manifest=args.manifest,
        audio_root=args.audio_root,
        output=args.output,
        max_per_class=args.max_per_class,
        max_per_speaker_class=args.max_per_speaker_class,
        seed=args.seed,
    )
    print(f"Сохранено: {args.output}")
    print(f"Записей: {len(result)}")
    print("По классам:", result["emotion"].value_counts().to_dict())
    print("Дикторов по классам:", result.groupby("emotion")["speaker_id"].nunique().to_dict())


if __name__ == "__main__":
    main()
