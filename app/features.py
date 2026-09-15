"""Извлечение и кэширование фиксированной схемы акустических признаков."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd

from .audio import load_audio, preprocess_signal
from .constants import FEATURE_SCHEMA_VERSION

logger = logging.getLogger(__name__)

try:
    import librosa
except Exception as exc:
    librosa = None
    logger.exception("librosa unavailable: %s", exc)

STATISTICS_PER_FEATURE = 4
FEATURE_COMPONENT_ROWS = {
    "mfcc": 13,
    "delta_mfcc": 13,
    "delta2_mfcc": 13,
    "chroma": 12,
    "spectral_centroid": 1,
    "spectral_bandwidth": 1,
    "spectral_rolloff": 1,
    "zcr": 1,
    "rms": 1,
    "mel": 40,
    "spectral_contrast": 7,
    "pitch": 1,
}
EXPECTED_FEATURE_COUNT = (
    sum(FEATURE_COMPONENT_ROWS.values()) * STATISTICS_PER_FEATURE
)


@dataclass(frozen=True)
class FeatureMatrix:

    feature_matrix: np.ndarray
    emotion_labels: np.ndarray
    audio_paths: list[str]
    feature_names: list[str]
    errors: list[str]
    speaker_ids: list[str]


def _summarize_feature_rows(
    values: np.ndarray, prefix: str
) -> tuple[list[float], list[str]]:
    feature_rows = np.asarray(values, dtype=np.float32)
    if feature_rows.ndim == 1:
        feature_rows = feature_rows.reshape(1, -1)
    feature_rows = np.nan_to_num(
        feature_rows, nan=0.0, posinf=0.0, neginf=0.0
    )
    statistics: list[float] = []
    statistic_names: list[str] = []
    for row_index, row_values in enumerate(feature_rows):
        row_statistics = {
            "mean": float(np.mean(row_values)),
            "std": float(np.std(row_values)),
            "min": float(np.min(row_values)),
            "max": float(np.max(row_values)),
        }
        for statistic_name, statistic_value in row_statistics.items():
            statistics.append(statistic_value)
            statistic_names.append(
                f"{prefix}_{row_index + 1}_{statistic_name}"
            )
    return statistics, statistic_names


class FeatureExtractor:

    def __init__(
        self,
        cache_dir: str | Path,
        sample_rate: int = 16_000,
        denoise: bool = True,
        trim: bool = True,
        normalize: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = int(sample_rate)
        self.denoise = bool(denoise)
        self.trim = bool(trim)
        self.normalize = bool(normalize)

    def _cache_key(self, path: str | Path) -> Path:
        audio_path = Path(path)
        file_metadata = audio_path.stat()
        payload = {
            "path": str(audio_path.resolve()),
            "mtime_ns": file_metadata.st_mtime_ns,
            "size": file_metadata.st_size,
            "sample_rate": self.sample_rate,
            "denoise": self.denoise,
            "trim": self.trim,
            "normalize": self.normalize,
            "version": FEATURE_SCHEMA_VERSION,
        }
        serialized_payload = json.dumps(payload, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(serialized_payload).hexdigest()
        return self.cache_dir / f"{digest}.joblib"

    def extract_file(self, path: str | Path) -> tuple[np.ndarray, list[str]]:
        cache_path = self._cache_key(path)
        if cache_path.exists():
            cached_features = joblib.load(cache_path)
            return cached_features["features"], cached_features["names"]

        audio_signal, sample_rate = load_audio(
            path, sample_rate=self.sample_rate, mono=True
        )
        processed_signal = preprocess_signal(
            audio_signal,
            normalize=self.normalize,
            denoise=self.denoise,
            trim=self.trim,
        )
        feature_vector, feature_names = self.extract_signal(
            processed_signal, sample_rate
        )
        joblib.dump(
            {"features": feature_vector, "names": feature_names}, cache_path
        )
        return feature_vector, feature_names

    @staticmethod
    def _extract_component(
        prefix: str,
        expected_rows: int,
        extractor: Callable[[], np.ndarray],
    ) -> tuple[list[float], list[str]]:
        try:
            values = np.asarray(extractor(), dtype=np.float32)
            if values.ndim == 1:
                values = values.reshape(1, -1)
            if values.shape[0] != expected_rows:
                raise ValueError(
                    f"ожидалось строк: {expected_rows}, получено: {values.shape[0]}"
                )
        except Exception as exc:
            logger.warning("Feature extraction failed for %s: %s", prefix, exc)
            values = np.zeros((expected_rows, 1), dtype=np.float32)
        return _summarize_feature_rows(values, prefix)

    @staticmethod
    def _mfcc_components(
        audio_signal: np.ndarray, sample_rate: int
    ) -> list[tuple[str, int, Callable[[], np.ndarray]]]:
        try:
            mfcc = librosa.feature.mfcc(
                y=audio_signal, sr=sample_rate, n_mfcc=13
            )
            return [
                ("mfcc", 13, lambda: mfcc),
                ("delta_mfcc", 13, lambda: librosa.feature.delta(mfcc)),
                (
                    "delta2_mfcc",
                    13,
                    lambda: librosa.feature.delta(mfcc, order=2),
                ),
            ]
        except Exception:
            logger.exception("MFCC extraction failed")
            return [
                ("mfcc", 13, lambda: np.zeros((13, 1))),
                ("delta_mfcc", 13, lambda: np.zeros((13, 1))),
                ("delta2_mfcc", 13, lambda: np.zeros((13, 1))),
            ]

    @staticmethod
    def _spectral_components(
        audio_signal: np.ndarray, sample_rate: int
    ) -> list[tuple[str, int, Callable[[], np.ndarray]]]:
        return [
            (
                "chroma",
                12,
                lambda: librosa.feature.chroma_stft(
                    y=audio_signal, sr=sample_rate
                ),
            ),
            (
                "spectral_centroid",
                1,
                lambda: librosa.feature.spectral_centroid(
                    y=audio_signal, sr=sample_rate
                ),
            ),
            (
                "spectral_bandwidth",
                1,
                lambda: librosa.feature.spectral_bandwidth(
                    y=audio_signal, sr=sample_rate
                ),
            ),
            (
                "spectral_rolloff",
                1,
                lambda: librosa.feature.spectral_rolloff(
                    y=audio_signal, sr=sample_rate
                ),
            ),
            (
                "zcr",
                1,
                lambda: librosa.feature.zero_crossing_rate(audio_signal),
            ),
            ("rms", 1, lambda: librosa.feature.rms(y=audio_signal)),
            (
                "mel",
                40,
                lambda: librosa.power_to_db(
                    librosa.feature.melspectrogram(
                        y=audio_signal, sr=sample_rate, n_mels=40
                    ),
                    ref=np.max,
                ),
            ),
            (
                "spectral_contrast",
                7,
                lambda: librosa.feature.spectral_contrast(
                    y=audio_signal, sr=sample_rate
                ),
            ),
            (
                "pitch",
                1,
                lambda: librosa.yin(
                    audio_signal, fmin=50, fmax=500, sr=sample_rate
                ),
            ),
        ]

    def extract_signal(
        self, audio_signal: np.ndarray, sample_rate: int
    ) -> tuple[np.ndarray, list[str]]:
        if librosa is None:
            raise RuntimeError("librosa не установлена.")
        prepared_signal = np.nan_to_num(
            np.asarray(audio_signal, dtype=np.float32)
        )
        if prepared_signal.size < 256:
            prepared_signal = np.pad(
                prepared_signal, (0, 256 - prepared_signal.size)
            )

        components = self._mfcc_components(
            prepared_signal, sample_rate
        ) + self._spectral_components(prepared_signal, sample_rate)

        feature_values: list[float] = []
        feature_names: list[str] = []
        for prefix, expected_rows, extractor in components:
            statistics, statistic_names = self._extract_component(
                prefix, expected_rows, extractor
            )
            feature_values.extend(statistics)
            feature_names.extend(statistic_names)

        if len(feature_values) != EXPECTED_FEATURE_COUNT:
            raise RuntimeError(
                "Нарушена внутренняя схема признаков: "
                f"ожидалось {EXPECTED_FEATURE_COUNT}, получено {len(feature_values)}."
            )
        return np.asarray(feature_values, dtype=np.float32), feature_names

    def build_matrix(self, dataframe: pd.DataFrame) -> FeatureMatrix:
        feature_vectors: list[np.ndarray] = []
        emotion_labels: list[str] = []
        audio_paths: list[str] = []
        extraction_errors: list[str] = []
        expected_feature_names: list[str] | None = None
        speaker_ids: list[str] = []

        for _, record in dataframe.iterrows():
            audio_path = str(record["file_path"])
            try:
                feature_vector, feature_names = self.extract_file(audio_path)
                if len(feature_vector) != EXPECTED_FEATURE_COUNT:
                    raise ValueError(
                        f"ожидалось {EXPECTED_FEATURE_COUNT} признаков, "
                        f"получено {len(feature_vector)}"
                    )
                if expected_feature_names is None:
                    expected_feature_names = feature_names
                elif feature_names != expected_feature_names:
                    raise ValueError("имена или порядок признаков не совпадают")
                feature_vectors.append(feature_vector)
                emotion_labels.append(str(record["emotion"]))
                audio_paths.append(audio_path)
                speaker_ids.append(
                    str(record.get("speaker_id", "unknown") or "unknown")
                )
            except Exception as exc:
                extraction_errors.append(f"{audio_path}: {exc}")

        if not feature_vectors or expected_feature_names is None:
            raise RuntimeError(
                "Не удалось извлечь признаки ни из одного аудиофайла."
            )
        return FeatureMatrix(
            feature_matrix=np.vstack(feature_vectors).astype(
                np.float32, copy=False
            ),
            emotion_labels=np.asarray(emotion_labels),
            audio_paths=audio_paths,
            feature_names=expected_feature_names,
            errors=extraction_errors,
            speaker_ids=speaker_ids,
        )
