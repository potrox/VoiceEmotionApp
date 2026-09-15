"""Формирование воспроизводимого идентификатора эксперимента."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from ..config import AppConfig
from ..constants import FEATURE_SCHEMA_VERSION
from ..dataset import DatasetLoadResult


def _hash_file_content(file_path: Path) -> str:
    content_hash = hashlib.sha256()
    with file_path.open("rb") as audio_file:
        for file_block in iter(lambda: audio_file.read(1024 * 1024), b""):
            content_hash.update(file_block)
    return content_hash.hexdigest()


def _file_fingerprint(path_value: str) -> dict[str, Any]:
    file_path = Path(path_value)
    try:
        file_metadata = file_path.stat()
    except OSError:
        return {"path": path_value, "size": None, "mtime_ns": None}
    return {
        "path": str(file_path.resolve()),
        "size": file_metadata.st_size,
        "mtime_ns": file_metadata.st_mtime_ns,
        "content_sha256": _hash_file_content(file_path),
    }


def build_experiment_signature(
    dataset_result: DatasetLoadResult,
    config: AppConfig,
    selected_models: Iterable[str],
    parameter_mode: str,
    manual_parameters: dict[str, Any],
) -> str:
    dataset_rows: list[dict[str, Any]] = []
    dataframe = dataset_result.dataframe
    if "file_path" in dataframe.columns:
        sort_columns = [
            column for column in ("file_path", "emotion") if column in dataframe.columns
        ]
        for _, row in dataframe.sort_values(by=sort_columns).iterrows():
            dataset_rows.append(
                {
                    "file": _file_fingerprint(str(row.get("file_path", ""))),
                    "emotion": str(row.get("emotion", "")),
                    "speaker_id": str(row.get("speaker_id", "")),
                }
            )

    signature_data = {
        "dataset_path": str(dataset_result.original_path),
        "records_count": int(len(dataframe)),
        "files": dataset_rows,
        "selected_models": sorted(selected_models),
        "parameter_mode": parameter_mode,
        "manual_parameters": manual_parameters if parameter_mode == "manual" else {},
        "training_config": {
            "test_size": config.test_size,
            "hyperparameter_mode": config.hyperparameter_mode,
            "feature_selection": config.feature_selection,
            "sample_rate": config.sample_rate,
            "denoise": config.denoise,
            "trim_silence": config.trim_silence,
            "normalize_amplitude": config.normalize_amplitude,
            "random_state": 42 if config.random_state is None else config.random_state,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        },
    }
    encoded_data = json.dumps(
        signature_data,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded_data).hexdigest()
