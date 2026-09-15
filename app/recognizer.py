"""Распознавание эмоций в аудиосигналах, файлах и папках."""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .audio import (
    analyze_quality,
    load_audio,
    preprocess_signal,
    save_wav,
    split_long_signal,
)
from .constants import EMOTION_RU, EMOTIONS
from .features import FeatureExtractor
from .ml.persistence import load_bundle, transform_features
from .microphone import MicrophoneRecorder
from .storage import timestamp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecognitionResult:

    file_path: str
    predicted_emotion: str
    predicted_emotion_ru: str
    confidence: float
    probabilities: dict[str, float]
    processing_time_sec: float
    model_name: str
    warnings: list[str]
    error: str = ""


@dataclass(frozen=True)
class SegmentRecognitionResult:

    index: int
    start_sec: float
    end_sec: float
    predicted_emotion: str
    predicted_emotion_ru: str
    confidence: float
    probabilities: dict[str, float]
    warnings: list[str]


@dataclass(frozen=True)
class SegmentedRecognitionResult:

    file_path: str
    predicted_emotion: str
    predicted_emotion_ru: str
    confidence: float
    probabilities: dict[str, float]
    processing_time_sec: float
    model_name: str
    warnings: list[str]
    segments: list[SegmentRecognitionResult]
    segment_duration_sec: float
    error: str = ""


class Recognizer:

    def __init__(self, model_path: str | Path, extractor: FeatureExtractor) -> None:
        self.model_path = Path(model_path)
        self.bundle = load_bundle(self.model_path)
        self.extractor = extractor

    def available_models(self) -> list[str]:
        return list(self.bundle.get("models", {}).keys())

    def _resolve_model_name(self, requested_model_name: str | None) -> str:
        if requested_model_name in (None, "", "Лучшая модель", "best"):
            model_name = self.bundle.get("best_model_name") or "Ансамбль SVM + MLP"
        else:
            model_name = requested_model_name
        if model_name not in self.bundle.get("models", {}):
            available_models = ", ".join(self.available_models())
            raise ValueError(
                f"Модель '{model_name}' отсутствует в сохранённом файле. "
                f"Доступно: {available_models}"
            )
        return model_name

    def _predict_matrix(
        self,
        raw_feature_matrix: np.ndarray,
        model_name: str | None = None,
    ) -> tuple[str, dict[str, float], str]:
        feature_matrix = transform_features(self.bundle, raw_feature_matrix)
        selected_model_name = self._resolve_model_name(model_name)
        model = self.bundle["models"][selected_model_name]
        label_encoder = self.bundle["label_encoder"]
        mean_probabilities = model.predict_proba(feature_matrix).mean(axis=0)
        predicted_class_index = int(np.argmax(mean_probabilities))
        predicted_emotion = str(
            label_encoder.inverse_transform([predicted_class_index])[0]
        )
        probabilities = {
            str(label_encoder.inverse_transform([class_index])[0]): float(probability)
            for class_index, probability in enumerate(mean_probabilities)
        }
        return predicted_emotion, probabilities, selected_model_name

    def _feature_matrix_from_signals(
        self, audio_signals: Iterable[np.ndarray], sample_rate: int
    ) -> np.ndarray:
        feature_vectors = [
            self.extractor.extract_signal(audio_signal, sample_rate)[0]
            for audio_signal in audio_signals
        ]
        if not feature_vectors:
            raise ValueError("Не передано ни одного аудиосигнала для распознавания.")
        expected_feature_count = len(self.bundle["feature_names"])
        invalid_feature_counts = sorted(
            {
                len(feature_vector)
                for feature_vector in feature_vectors
                if len(feature_vector) != expected_feature_count
            }
        )
        if invalid_feature_counts:
            raise ValueError(
                "Схема признаков модели не совпадает с текущим экстрактором: "
                f"ожидалось {expected_feature_count}, получено "
                + ", ".join(map(str, invalid_feature_counts))
                + ". Переобучите модель текущей версией приложения."
            )
        return np.vstack(feature_vectors).astype(np.float32, copy=False)

    def _vector_from_signal(
        self, audio_signal: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        return self._feature_matrix_from_signals([audio_signal], sample_rate)

    def recognize_file(
        self, path: str | Path, model_name: str | None = None
    ) -> RecognitionResult:
        audio_path = Path(path)
        started_at = time.perf_counter()
        warnings: list[str] = []
        try:
            audio_signal, sample_rate = load_audio(
                audio_path, sample_rate=self.extractor.sample_rate, mono=True
            )
            processed_signal = preprocess_signal(
                audio_signal,
                normalize=self.extractor.normalize,
                denoise=self.extractor.denoise,
                trim=self.extractor.trim,
            )
            quality = analyze_quality(processed_signal, sample_rate, training=False)
            warnings.extend(quality.warnings)
            signal_parts = split_long_signal(
                processed_signal, sample_rate, max_duration_sec=10.0
            )
            raw_feature_matrix = self._feature_matrix_from_signals(
                signal_parts, sample_rate
            )
            predicted_emotion, probabilities, selected_model_name = (
                self._predict_matrix(raw_feature_matrix, model_name=model_name)
            )
            return RecognitionResult(
                file_path=str(audio_path),
                predicted_emotion=predicted_emotion,
                predicted_emotion_ru=EMOTION_RU.get(
                    predicted_emotion, predicted_emotion
                ),
                confidence=float(probabilities[predicted_emotion]),
                probabilities=probabilities,
                processing_time_sec=float(time.perf_counter() - started_at),
                model_name=selected_model_name,
                warnings=warnings,
            )
        except Exception as exc:
            logger.exception("Recognition failed for %s", audio_path)
            return RecognitionResult(
                file_path=str(audio_path),
                predicted_emotion="",
                predicted_emotion_ru="",
                confidence=0.0,
                probabilities={},
                processing_time_sec=float(time.perf_counter() - started_at),
                model_name=model_name or "",
                warnings=warnings,
                error=str(exc),
            )

    @staticmethod
    def _split_signal_into_segments(
        audio_signal: np.ndarray,
        sample_rate: int,
        segment_duration_seconds: float,
    ) -> list[tuple[int, int, np.ndarray]]:
        segment_length = max(1, int(segment_duration_seconds * sample_rate))
        segments: list[tuple[int, int, np.ndarray]] = []
        for start_index in range(0, len(audio_signal), segment_length):
            end_index = min(start_index + segment_length, len(audio_signal))
            segment_signal = audio_signal[start_index:end_index]
            is_short_tail = (
                len(audio_signal) > segment_length
                and segment_signal.size < int(0.5 * sample_rate)
            )
            if segment_signal.size and not is_short_tail:
                segments.append((start_index, end_index, segment_signal))
        return segments or [(0, len(audio_signal), audio_signal)]

    def _recognize_segment(
        self,
        segment_number: int,
        start_index: int,
        end_index: int,
        segment_signal: np.ndarray,
        sample_rate: int,
        model_name: str | None,
    ) -> tuple[SegmentRecognitionResult, str]:
        processed_signal = preprocess_signal(
            segment_signal,
            normalize=self.extractor.normalize,
            denoise=self.extractor.denoise,
            trim=self.extractor.trim,
        )
        quality = analyze_quality(
            processed_signal, sample_rate, training=False, min_duration_sec=0.0
        )
        segment_warnings = list(quality.warnings)
        if processed_signal.size < int(0.25 * sample_rate):
            processed_signal = preprocess_signal(
                segment_signal,
                normalize=self.extractor.normalize,
                denoise=self.extractor.denoise,
                trim=False,
            )
            segment_warnings.append(
                "После удаления тишины фрагмент стал слишком коротким, "
                "использован вариант без trim."
            )
        predicted_emotion, probabilities, selected_model_name = self._predict_matrix(
            self._vector_from_signal(processed_signal, sample_rate),
            model_name=model_name,
        )
        return (
            SegmentRecognitionResult(
                index=segment_number,
                start_sec=float(start_index / sample_rate),
                end_sec=float(end_index / sample_rate),
                predicted_emotion=predicted_emotion,
                predicted_emotion_ru=EMOTION_RU.get(
                    predicted_emotion, predicted_emotion
                ),
                confidence=float(probabilities.get(predicted_emotion, 0.0)),
                probabilities=probabilities,
                warnings=segment_warnings,
            ),
            selected_model_name,
        )

    @staticmethod
    def _average_probabilities(
        segment_results: list[SegmentRecognitionResult],
    ) -> dict[str, float]:
        emotion_labels = sorted(
            set().union(
                *(segment.probabilities.keys() for segment in segment_results)
            )
        )
        return {
            emotion: float(
                np.mean(
                    [
                        segment.probabilities.get(emotion, 0.0)
                        for segment in segment_results
                    ]
                )
            )
            for emotion in emotion_labels
        }

    def recognize_signal_segmented(
        self,
        audio_signal: np.ndarray,
        sample_rate: int,
        source_label: str,
        model_name: str | None = None,
        segment_duration_sec: float = 5.0,
    ) -> SegmentedRecognitionResult:
        started_at = time.perf_counter()
        warnings: list[str] = []
        segment_results: list[SegmentRecognitionResult] = []
        try:
            audio_signal = np.asarray(audio_signal, dtype=np.float32)
            if audio_signal.size == 0:
                raise RuntimeError("Запись пустая: микрофон не передал аудиоданные.")
            if sample_rate <= 0:
                raise ValueError("Частота дискретизации должна быть больше нуля.")
            recording_duration = float(len(audio_signal) / sample_rate)
            if recording_duration < 1.0:
                warnings.append("Запись короче 1 секунды. Итог может быть ненадёжным.")

            selected_model_name = self._resolve_model_name(model_name)
            raw_segments = self._split_signal_into_segments(
                audio_signal, sample_rate, float(segment_duration_sec)
            )
            for segment_number, (start_index, end_index, segment_signal) in enumerate(
                raw_segments, start=1
            ):
                segment_result, selected_model_name = self._recognize_segment(
                    segment_number,
                    start_index,
                    end_index,
                    segment_signal,
                    sample_rate,
                    selected_model_name,
                )
                segment_results.append(segment_result)

            overall_probabilities = self._average_probabilities(segment_results)
            if not overall_probabilities:
                raise RuntimeError(
                    "Не удалось получить вероятности ни для одного фрагмента записи."
                )
            predicted_emotion = max(
                overall_probabilities, key=overall_probabilities.get
            )
            if len(segment_results) == 1 and recording_duration < segment_duration_sec:
                warnings.append(
                    f"Запись короче {segment_duration_sec:.0f} секунд, поэтому "
                    "был обработан один неполный фрагмент."
                )
            return SegmentedRecognitionResult(
                file_path=str(source_label),
                predicted_emotion=predicted_emotion,
                predicted_emotion_ru=EMOTION_RU.get(
                    predicted_emotion, predicted_emotion
                ),
                confidence=float(overall_probabilities[predicted_emotion]),
                probabilities=overall_probabilities,
                processing_time_sec=float(time.perf_counter() - started_at),
                model_name=selected_model_name,
                warnings=warnings,
                segments=segment_results,
                segment_duration_sec=float(segment_duration_sec),
            )
        except Exception as exc:
            logger.exception("Segmented recognition failed for %s", source_label)
            return SegmentedRecognitionResult(
                file_path=str(source_label),
                predicted_emotion="",
                predicted_emotion_ru="",
                confidence=0.0,
                probabilities={},
                processing_time_sec=float(time.perf_counter() - started_at),
                model_name=model_name or "",
                warnings=warnings,
                segments=segment_results,
                segment_duration_sec=float(segment_duration_sec),
                error=str(exc),
            )

    def recognize_file_segmented(
        self,
        path: str | Path,
        model_name: str | None = None,
        segment_duration_sec: float = 5.0,
    ) -> SegmentedRecognitionResult:
        audio_path = Path(path)
        try:
            audio_signal, sample_rate = load_audio(
                audio_path, sample_rate=self.extractor.sample_rate, mono=True
            )
            return self.recognize_signal_segmented(
                audio_signal,
                sample_rate,
                str(audio_path),
                model_name=model_name,
                segment_duration_sec=segment_duration_sec,
            )
        except Exception as exc:
            logger.exception("Segmented file recognition failed for %s", audio_path)
            return SegmentedRecognitionResult(
                file_path=str(audio_path),
                predicted_emotion="",
                predicted_emotion_ru="",
                confidence=0.0,
                probabilities={},
                processing_time_sec=0.0,
                model_name=model_name or "",
                warnings=[],
                segments=[],
                segment_duration_sec=float(segment_duration_sec),
                error=str(exc),
            )

    def recognize_microphone(
        self,
        duration_sec: int = 5,
        model_name: str | None = None,
        device_index: int | None = None,
    ) -> RecognitionResult:
        audio_signal = MicrophoneRecorder(
            self.extractor.sample_rate, device_index=device_index
        ).record(duration_sec).audio_signal
        temporary_path = (
            self.extractor.cache_dir / f"mic_recognition_{timestamp()}.wav"
        )
        save_wav(temporary_path, audio_signal, self.extractor.sample_rate)
        return self.recognize_file(temporary_path, model_name=model_name)

    @staticmethod
    def _recognition_csv_row(result: RecognitionResult) -> dict[str, str]:
        row = {
            "file_path": result.file_path,
            "predicted_emotion": result.predicted_emotion,
            "predicted_emotion_ru": result.predicted_emotion_ru,
            "confidence": f"{result.confidence:.6f}",
            "processing_time_sec": f"{result.processing_time_sec:.6f}",
            "model_name": result.model_name,
            "warnings": "; ".join(result.warnings),
            "error": result.error,
        }
        row.update(
            {
                f"prob_{emotion}": f"{result.probabilities.get(emotion, 0.0):.6f}"
                for emotion in EMOTIONS
            }
        )
        return row

    def batch_recognize(
        self,
        folder: str | Path,
        exports_dir: str | Path,
        model_name: str | None = None,
    ) -> Path:
        audio_folder = Path(folder)
        export_folder = Path(exports_dir)
        export_folder.mkdir(parents=True, exist_ok=True)
        audio_paths = sorted(
            list(audio_folder.rglob("*.wav")) + list(audio_folder.rglob("*.mp3"))
        )
        operation_timestamp = timestamp()
        csv_path = export_folder / f"recognition_results_{operation_timestamp}.csv"
        field_names = [
            "file_path", "predicted_emotion", "predicted_emotion_ru", "confidence",
            *[f"prob_{emotion}" for emotion in EMOTIONS],
            "processing_time_sec", "model_name", "warnings", "error",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            writer.writeheader()
            for audio_path in audio_paths:
                writer.writerow(
                    self._recognition_csv_row(
                        self.recognize_file(audio_path, model_name=model_name)
                    )
                )
        summary_path = export_folder / f"recognition_results_{operation_timestamp}.txt"
        summary_path.write_text(
            f"Пакетное распознавание завершено. Обработано файлов: {len(audio_paths)}\n"
            f"CSV: {csv_path}\n",
            encoding="utf-8",
        )
        return csv_path
