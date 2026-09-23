from __future__ import annotations

import csv
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .asr import LocalTranscriber, Transcription
from .audio import analyze_quality, load_audio, preprocess_signal, record_microphone, save_wav, split_long_signal
from .constants import EMOTION_RU, EMOTIONS
from .features import FeatureExtractor
from .embedding_features import Emotion2VecExtractor
from .models import load_bundle, transform_features
from .storage import timestamp

logger = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    file_path: str
    predicted_emotion: str
    predicted_emotion_ru: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time_sec: float
    model_name: str
    warnings: List[str]
    error: str = ""
    transcript: str = ""
    transcript_source: str = ""


@dataclass
class SegmentRecognitionResult:
    index: int
    start_sec: float
    end_sec: float
    predicted_emotion: str
    predicted_emotion_ru: str
    confidence: float
    probabilities: Dict[str, float]
    warnings: List[str]


@dataclass
class SegmentedRecognitionResult:
    file_path: str
    predicted_emotion: str
    predicted_emotion_ru: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time_sec: float
    model_name: str
    warnings: List[str]
    segments: List[SegmentRecognitionResult]
    segment_duration_sec: float
    error: str = ""
    transcript: str = ""
    transcript_source: str = ""


class Recognizer:
    def __init__(self, model_path: str | Path, extractor: FeatureExtractor):
        self.model_path = Path(model_path)
        self.bundle = load_bundle(self.model_path)
        params = self.bundle.get("preprocessing_params", {})
        self.extractor = FeatureExtractor(
            extractor.cache_dir,
            sample_rate=int(params.get("sample_rate", extractor.sample_rate)),
            denoise=bool(params.get("denoise", extractor.denoise)),
            trim=bool(params.get("trim_silence", extractor.trim)),
            normalize=bool(params.get("normalize_amplitude", extractor.normalize)),
        )
        self.feature_backend = str(params.get("feature_backend", "handcrafted"))
        self.reliability_threshold = float(params.get("reliability_threshold", 0.0))
        self.transcriber: LocalTranscriber | None = None
        self._transcriber_lock = threading.RLock()
        self.embedding_extractor: Emotion2VecExtractor | None = None
        if self.feature_backend == "emotion2vec":
            self.embedding_extractor = Emotion2VecExtractor(
                model_id=str(params.get("emotion2vec_model", "emotion2vec/emotion2vec_plus_base")),
                dimension=int(params.get("embedding_dimension", 768)),
                ncpu=int(params.get("embedding_ncpu", 4)),
            )

    def _add_reliability_warning(self, warnings: List[str], confidence: float) -> None:
        if self.reliability_threshold > 0 and confidence < self.reliability_threshold:
            warnings.append(
                "Низкая уверенность модели: результат лучше считать предположением, "
                "а не надёжной автоматической меткой."
            )

    def _transcribe_file(self, path: Path) -> Transcription:
        with self._transcriber_lock:
            if self.transcriber is None:
                self.transcriber = LocalTranscriber(self.extractor.cache_dir)
            transcriber = self.transcriber
        return transcriber.transcribe_file(path)

    def available_models(self) -> List[str]:
        return list(self.bundle.get("models", {}).keys())

    def _predict_matrix(
        self,
        X_raw: np.ndarray,
        model_name: str | None = None,
        transcript: str | None = None,
    ) -> Tuple[str, Dict[str, float], str]:
        X = transform_features(self.bundle, X_raw)
        if model_name in (None, "", "Лучшая модель", "best"):
            name = self.bundle.get("best_model_name") or "Logistic Regression"
        else:
            name = model_name
        if name not in self.bundle.get("models", {}):
            available = ", ".join(self.available_models())
            raise ValueError(f"Модель '{name}' отсутствует в сохранённом файле. Доступно: {available}")
        model = self.bundle["models"][name]
        label_encoder = self.bundle["label_encoder"]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
        elif hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(X), dtype=float)
            if scores.ndim == 1:
                scores = np.column_stack([-scores, scores])
            scores = scores - scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        else:
            raise ValueError(f"Модель '{name}' не поддерживает оценку классов.")
        mean_probs = probs.mean(axis=0)
        clean_transcript = str(transcript or "").strip()
        text_model = self.bundle.get("text_model")
        if clean_transcript and text_model is not None:
            text_probs = np.asarray(text_model.predict_proba([clean_transcript])[0], dtype=float)
            if text_probs.shape != mean_probs.shape:
                raise ValueError(
                    "Размерность вероятностей текстовой модели не совпадает с аудиомоделью."
                )
            audio_weight = float(self.bundle.get("fusion_audio_weight", 0.4))
            if not 0.0 <= audio_weight <= 1.0:
                raise ValueError("Вес аудиомодели должен находиться в диапазоне [0, 1].")
            mean_probs = audio_weight * mean_probs + (1.0 - audio_weight) * text_probs
            name = f"{name} + транскрипт"
        idx = int(np.argmax(mean_probs))
        label = str(label_encoder.inverse_transform([idx])[0])
        prob_map = {str(label_encoder.inverse_transform([i])[0]): float(mean_probs[i]) for i in range(len(mean_probs))}
        return label, prob_map, name

    def _vector_from_signal(self, y: np.ndarray, sr: int) -> np.ndarray:
        if self.embedding_extractor is not None:
            vec = self.embedding_extractor.extract_signal(y, sr)
        else:
            vec, _ = self.extractor.extract_signal(y, sr)
        max_len = len(self.bundle["feature_names"])
        X_raw = np.zeros((1, max_len), dtype=np.float32)
        X_raw[0, :min(len(vec), max_len)] = vec[:max_len]
        return X_raw

    def recognize_file(
        self,
        path: str | Path,
        model_name: str | None = None,
        transcript: str | None = None,
        auto_transcribe: bool = True,
    ) -> RecognitionResult:
        path = Path(path)
        t0 = time.perf_counter()
        warnings: List[str] = []
        final_transcript = str(transcript or "").strip()
        transcript_source = "ручной" if final_transcript else ""
        try:
            y, sr = load_audio(path, sample_rate=self.extractor.sample_rate, mono=True)
            y = preprocess_signal(y, sr, normalize=self.extractor.normalize, denoise=self.extractor.denoise, trim=self.extractor.trim)
            q = analyze_quality(y, sr, training=False)
            warnings.extend(q.warnings)
            if self.embedding_extractor is not None:
                vectors = [self.embedding_extractor.extract_file(path)]
            else:
                chunks = split_long_signal(y, sr, max_duration_sec=10.0)
                vectors = []
                for chunk in chunks:
                    vec, _ = self.extractor.extract_signal(chunk, sr)
                    vectors.append(vec)
            max_len = len(self.bundle["feature_names"])
            X_raw = np.zeros((len(vectors), max_len), dtype=np.float32)
            for i, vec in enumerate(vectors):
                X_raw[i, :min(len(vec), max_len)] = vec[:max_len]
            if not final_transcript and auto_transcribe and self.bundle.get("text_model") is not None:
                try:
                    final_transcript = self._transcribe_file(path).text
                    transcript_source = "автоматический" if final_transcript else ""
                    if not final_transcript:
                        warnings.append("ASR не обнаружил речь: использована только аудиомодель.")
                except Exception as exc:
                    logger.warning("ASR failed for %s: %s", path, exc)
                    warnings.append(f"Автоматическая транскрипция недоступна ({exc}); использована только аудиомодель.")
            label, prob_map, used_model = self._predict_matrix(
                X_raw, model_name=model_name, transcript=final_transcript
            )
            self._add_reliability_warning(warnings, float(prob_map[label]))
            elapsed = time.perf_counter() - t0
            return RecognitionResult(
                file_path=str(path),
                predicted_emotion=label,
                predicted_emotion_ru=EMOTION_RU.get(label, label),
                confidence=float(prob_map[label]),
                probabilities=prob_map,
                processing_time_sec=float(elapsed),
                model_name=used_model,
                warnings=warnings,
                transcript=final_transcript,
                transcript_source=transcript_source,
            )
        except Exception as exc:
            logger.exception("Recognition failed for %s", path)
            return RecognitionResult(
                file_path=str(path), predicted_emotion="", predicted_emotion_ru="", confidence=0.0,
                probabilities={}, processing_time_sec=float(time.perf_counter() - t0), model_name=model_name or "", warnings=warnings, error=str(exc),
                transcript=final_transcript, transcript_source=transcript_source,
            )

    def recognize_signal_segmented(
        self,
        y: np.ndarray,
        sr: int,
        source_label: str,
        model_name: str | None = None,
        segment_duration_sec: float = 5.0,
        transcription: Transcription | None = None,
    ) -> SegmentedRecognitionResult:
        """Recognize a long microphone/file recording by fixed-length fragments.

        Used for the microphone recognition mode where the user manually stops
        recording. The full recording is split into 5-second fragments, each
        fragment receives its own prediction, and the overall emotional background
        is calculated as the mean probability distribution over all fragments.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []
        segments: List[SegmentRecognitionResult] = []
        try:
            y = np.asarray(y, dtype=np.float32)
            if y.size == 0:
                raise RuntimeError("Запись пустая: микрофон не передал аудиоданные.")
            duration = float(len(y) / sr) if sr else 0.0
            if duration < 1.0:
                warnings.append("Запись короче 1 секунды. Итог может быть ненадёжным.")

            segment_len = max(1, int(float(segment_duration_sec) * int(sr)))
            raw_segments: List[tuple[int, int, np.ndarray]] = []
            for start in range(0, len(y), segment_len):
                end = min(start + segment_len, len(y))
                chunk = y[start:end]
                if chunk.size == 0:
                    continue


                if len(y) > segment_len and chunk.size < int(0.5 * sr):
                    continue
                raw_segments.append((start, end, chunk))
            if not raw_segments:
                raw_segments.append((0, len(y), y))

            model_used: str | None = None
            used_names: set[str] = set()
            prob_vectors: List[Dict[str, float]] = []
            for idx, (start, end, chunk) in enumerate(raw_segments, start=1):
                chunk_warnings: List[str] = []
                processed = preprocess_signal(
                    chunk,
                    sr,
                    normalize=self.extractor.normalize,
                    denoise=self.extractor.denoise,
                    trim=self.extractor.trim,
                )
                q = analyze_quality(processed, sr, training=False, min_duration_sec=0.0)
                chunk_warnings.extend(q.warnings)
                if processed.size < int(0.25 * sr):



                    processed = preprocess_signal(
                        chunk,
                        sr,
                        normalize=self.extractor.normalize,
                        denoise=self.extractor.denoise,
                        trim=False,
                    )
                    chunk_warnings.append("После удаления тишины фрагмент стал слишком коротким, использован вариант без trim.")
                X_raw = self._vector_from_signal(processed, sr)
                chunk_text = transcription.text_for_span(start / sr, end / sr) if transcription else ""
                label, prob_map, used_name = self._predict_matrix(
                    X_raw, model_name=model_name, transcript=chunk_text
                )
                model_used = used_name
                used_names.add(used_name)
                prob_vectors.append(prob_map)
                confidence = float(prob_map.get(label, 0.0))
                self._add_reliability_warning(chunk_warnings, confidence)
                segments.append(SegmentRecognitionResult(
                    index=idx,
                    start_sec=float(start / sr),
                    end_sec=float(end / sr),
                    predicted_emotion=label,
                    predicted_emotion_ru=EMOTION_RU.get(label, label),
                    confidence=confidence,
                    probabilities=prob_map,
                    warnings=chunk_warnings,
                ))

            if not prob_vectors:
                raise RuntimeError("Не удалось получить вероятности ни для одного фрагмента записи.")
            all_labels = sorted(set().union(*(p.keys() for p in prob_vectors)))
            overall_probs = {label: float(np.mean([p.get(label, 0.0) for p in prob_vectors])) for label in all_labels}
            predicted = max(overall_probs, key=overall_probs.get)
            self._add_reliability_warning(warnings, float(overall_probs[predicted]))
            elapsed = time.perf_counter() - t0
            if len(segments) == 1 and duration < float(segment_duration_sec):
                warnings.append(
                    f"Запись короче {float(segment_duration_sec):.0f} секунд, поэтому был обработан один неполный фрагмент."
                )
            return SegmentedRecognitionResult(
                file_path=str(source_label),
                predicted_emotion=predicted,
                predicted_emotion_ru=EMOTION_RU.get(predicted, predicted),
                confidence=float(overall_probs[predicted]),
                probabilities=overall_probs,
                processing_time_sec=float(elapsed),
                model_name=("Смешанный режим: аудио и транскрипт по фрагментам" if len(used_names) > 1 else (model_used or model_name or "")),
                warnings=warnings,
                segments=segments,
                segment_duration_sec=float(segment_duration_sec),
                transcript=transcription.text if transcription else "",
                transcript_source="автоматический" if transcription and transcription.text else "",
            )
        except Exception as exc:
            logger.exception("Segmented recognition failed for %s", source_label)
            return SegmentedRecognitionResult(
                file_path=str(source_label), predicted_emotion="", predicted_emotion_ru="", confidence=0.0,
                probabilities={}, processing_time_sec=float(time.perf_counter() - t0), model_name=model_name or "",
                warnings=warnings, segments=segments, segment_duration_sec=float(segment_duration_sec), error=str(exc),
            )

    def recognize_file_segmented(
        self,
        path: str | Path,
        model_name: str | None = None,
        segment_duration_sec: float = 5.0,
        auto_transcribe: bool = True,
    ) -> SegmentedRecognitionResult:
        path = Path(path)
        started = time.perf_counter()
        try:
            y, sr = load_audio(path, sample_rate=self.extractor.sample_rate, mono=True)
            transcription = None
            asr_warning = ""
            if auto_transcribe and self.bundle.get("text_model") is not None:
                try:
                    transcription = self._transcribe_file(path)
                    if not transcription.text:
                        asr_warning = "ASR не обнаружил речь: использована только аудиомодель."
                except Exception as exc:
                    logger.warning("ASR failed for %s: %s", path, exc)
                    asr_warning = f"Автоматическая транскрипция недоступна ({exc}); использована только аудиомодель."
            result = self.recognize_signal_segmented(
                y, sr, str(path), model_name=model_name,
                segment_duration_sec=segment_duration_sec, transcription=transcription,
            )
            if asr_warning:
                result.warnings.append(asr_warning)
            result.processing_time_sec = float(time.perf_counter() - started)
            return result
        except Exception as exc:
            logger.exception("Segmented file recognition failed for %s", path)
            return SegmentedRecognitionResult(
                file_path=str(path), predicted_emotion="", predicted_emotion_ru="", confidence=0.0,
                probabilities={}, processing_time_sec=float(time.perf_counter() - started), model_name=model_name or "", warnings=[], segments=[],
                segment_duration_sec=float(segment_duration_sec), error=str(exc),
            )

    def recognize_microphone(self, duration_sec: int = 5, model_name: str | None = None, device_index: int | None = None) -> RecognitionResult:
        y = record_microphone(duration_sec, self.extractor.sample_rate, device_index=device_index)
        temp_path = self.extractor.cache_dir / f"mic_recognition_{timestamp()}.wav"
        save_wav(temp_path, y, self.extractor.sample_rate)
        return self.recognize_file(temp_path, model_name=model_name)

    def batch_recognize(self, folder: str | Path, exports_dir: str | Path, model_name: str | None = None, auto_transcribe: bool = True) -> Path:
        folder = Path(folder)
        exports_dir = Path(exports_dir)
        exports_dir.mkdir(parents=True, exist_ok=True)
        paths = sorted(list(folder.rglob("*.wav")) + list(folder.rglob("*.mp3")))
        out_path = exports_dir / f"recognition_results_{timestamp()}.csv"
        fieldnames = ["file_path", "predicted_emotion", "predicted_emotion_ru", "confidence"] + [f"prob_{e}" for e in EMOTIONS] + ["processing_time_sec", "model_name", "transcript", "transcript_source", "warnings", "error"]
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for path in paths:
                result = self.recognize_file(path, model_name=model_name, auto_transcribe=auto_transcribe)
                row = {
                    "file_path": result.file_path,
                    "predicted_emotion": result.predicted_emotion,
                    "predicted_emotion_ru": result.predicted_emotion_ru,
                    "confidence": f"{result.confidence:.6f}",
                    "processing_time_sec": f"{result.processing_time_sec:.6f}",
                    "model_name": result.model_name,
                    "transcript": result.transcript,
                    "transcript_source": result.transcript_source,
                    "warnings": "; ".join(result.warnings),
                    "error": result.error,
                }
                for emotion in EMOTIONS:
                    row[f"prob_{emotion}"] = f"{result.probabilities.get(emotion, 0.0):.6f}"
                writer.writerow(row)
        txt_path = exports_dir / f"recognition_results_{timestamp()}.txt"
        txt_path.write_text(f"Пакетное распознавание завершено. Обработано файлов: {len(paths)}\nCSV: {out_path}\n", encoding="utf-8")
        return out_path
