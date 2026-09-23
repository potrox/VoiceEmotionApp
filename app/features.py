from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd

from .audio import load_audio, preprocess_signal

logger = logging.getLogger(__name__)

try:
    import librosa
except Exception as exc:
    librosa = None
    logger.exception("librosa unavailable: %s", exc)


@dataclass
class FeatureMatrix:
    X: np.ndarray
    y: np.ndarray
    paths: List[str]
    feature_names: List[str]
    errors: List[str]
    speaker_ids: List[str]
    split_labels: List[str]


def _stats(values: np.ndarray, prefix: str) -> Tuple[List[float], List[str]]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    feats: List[float] = []
    names: List[str] = []
    for i, row in enumerate(arr):
        # Mean and standard deviation retain the global contour of each
        # descriptor while halving the feature vector compared with the legacy
        # mean/std/min/max profile.  Min/max were unstable on short noisy clips
        # and encouraged overfitting on small datasets.
        for stat_name, stat_value in [
            ("mean", float(np.mean(row))),
            ("std", float(np.std(row))),
        ]:
            feats.append(stat_value)
            names.append(f"{prefix}_{i + 1}_{stat_name}")
    return feats, names


class FeatureExtractor:
    def __init__(self, cache_dir: str | Path, sample_rate: int = 16000, denoise: bool = True, trim: bool = True, normalize: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.denoise = denoise
        self.trim = trim
        self.normalize = normalize

    def _cache_key(self, path: str | Path) -> Path:
        p = Path(path)
        stat = p.stat()
        payload = {
            "path": str(p.resolve()),
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "sample_rate": self.sample_rate,
            "denoise": self.denoise,
            "trim": self.trim,
            "normalize": self.normalize,
            "version": 3,
            "feature_profile": "compact_v1",
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.joblib"

    def extract_file(self, path: str | Path) -> Tuple[np.ndarray, List[str]]:
        cache_path = self._cache_key(path)
        if cache_path.exists():
            data = joblib.load(cache_path)
            return data["features"], data["names"]
        y, sr = load_audio(path, sample_rate=self.sample_rate, mono=True)
        y = preprocess_signal(y, sr, normalize=self.normalize, denoise=self.denoise, trim=self.trim)
        feats, names = self.extract_signal(y, sr)
        joblib.dump({"features": feats, "names": names}, cache_path)
        return feats, names

    def extract_signal(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, List[str]]:
        if librosa is None:
            raise RuntimeError("librosa не установлена.")
        y = np.asarray(y, dtype=np.float32)
        y = np.nan_to_num(y)
        if y.size < 256:
            y = np.pad(y, (0, 256 - y.size))
        features: List[float] = []
        names: List[str] = []

        def add(values: np.ndarray, prefix: str) -> None:
            f, n = _stats(values, prefix)
            features.extend(f)
            names.extend(n)

        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            add(mfcc, "mfcc")
            add(librosa.feature.delta(mfcc), "delta_mfcc")
            add(librosa.feature.delta(mfcc, order=2), "delta2_mfcc")
        except Exception:
            logger.exception("MFCC extraction failed")
            add(np.zeros((13, 1)), "mfcc")
            add(np.zeros((13, 1)), "delta_mfcc")
            add(np.zeros((13, 1)), "delta2_mfcc")

        # Compact speech-focused profile.  Chroma, tonnetz, tempo and the full
        # mel summary were removed: they were expensive, highly correlated with
        # the MFCC block and weakly justified for short emotional speech clips.
        feature_calls = [
            ("spectral_centroid", lambda: librosa.feature.spectral_centroid(y=y, sr=sr)),
            ("spectral_bandwidth", lambda: librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            ("spectral_rolloff", lambda: librosa.feature.spectral_rolloff(y=y, sr=sr)),
            ("zcr", lambda: librosa.feature.zero_crossing_rate(y)),
            ("rms", lambda: librosa.feature.rms(y=y)),
            ("spectral_contrast", lambda: librosa.feature.spectral_contrast(y=y, sr=sr)),
        ]
        for prefix, func in feature_calls:
            try:
                values = func()
                add(values, prefix)
            except Exception:
                logger.exception("Feature extraction failed: %s", prefix)
                add(np.zeros((1, 1)), prefix)

        try:
            f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
            f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
            add(f0.reshape(1, -1), "pitch")
        except Exception:
            logger.exception("Pitch extraction failed")
            add(np.zeros((1, 1)), "pitch")

        return np.asarray(features, dtype=np.float32), names

    def build_matrix(
        self,
        df: pd.DataFrame,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> FeatureMatrix:
        xs: List[np.ndarray] = []
        ys: List[str] = []
        paths: List[str] = []
        errors: List[str] = []
        feature_names: List[str] = []
        speaker_ids: List[str] = []
        split_labels: List[str] = []
        total = len(df)
        for position, (_, row) in enumerate(df.iterrows(), start=1):
            path = str(row["file_path"])
            try:
                vec, names = self.extract_file(path)
                if not feature_names:
                    feature_names = names
                xs.append(vec)
                ys.append(str(row["emotion"]))
                paths.append(path)
                speaker_ids.append(str(row.get("speaker_id", "unknown") or "unknown"))
                split_labels.append(str(row.get("dataset_split", "") or "").strip().lower())
            except Exception as exc:
                errors.append(f"{path}: {exc}")
            if progress_callback and (position == 1 or position % 100 == 0 or position == total):
                progress_callback(position, total)
        if not xs:
            raise RuntimeError("Не удалось извлечь признаки ни из одного аудиофайла.")

        max_len = max(len(x) for x in xs)
        X = np.zeros((len(xs), max_len), dtype=np.float32)
        for i, vec in enumerate(xs):
            X[i, :len(vec)] = vec
        if len(feature_names) < max_len:
            feature_names += [f"feature_{i}" for i in range(len(feature_names), max_len)]
        return FeatureMatrix(
            X=X,
            y=np.asarray(ys),
            paths=paths,
            feature_names=feature_names[:max_len],
            errors=errors,
            speaker_ids=speaker_ids,
            split_labels=split_labels,
        )
