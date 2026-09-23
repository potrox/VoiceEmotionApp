"""Local, lazy Russian speech transcription for emotion recognition."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ASRSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcription:
    text: str
    segments: tuple[ASRSegment, ...]

    def text_for_span(self, start: float, end: float) -> str:
        return " ".join(
            segment.text for segment in self.segments
            if max(0.0, min(segment.end, end) - max(segment.start, start))
            >= 0.5 * max(0.001, segment.end - segment.start)
        ).strip()


class LocalTranscriber:
    """Use a frozen faster-whisper model on CPU; audio never leaves this machine."""

    def __init__(self, cache_dir: str | Path, model_size: str = "small", cpu_threads: int = 4):
        self.cache_dir = Path(cache_dir) / "asr"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_size = model_size
        self.cpu_threads = max(1, int(cpu_threads))
        self._model = None
        self._lock = threading.RLock()

    def _ensure_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "Для автоматической транскрипции установите faster-whisper из requirements.txt."
                ) from exc
            model_source = self.model_size
            # Also accept a fully downloaded local snapshot when Hugging Face's
            # Windows cache cannot create symlinks (or its transfer stalls).
            snapshots = self.cache_dir / "models" / f"models--Systran--faster-whisper-{self.model_size}" / "snapshots"
            if self.model_size == "small" and snapshots.exists():
                for candidate in snapshots.iterdir():
                    weights = candidate / "model.bin"
                    if weights.is_file() and weights.stat().st_size >= 450_000_000 and (candidate / "config.json").is_file():
                        model_source = str(candidate)
                        break
            self._model = WhisperModel(
                model_source,
                device="cpu",
                compute_type="int8",
                cpu_threads=self.cpu_threads,
                download_root=str(self.cache_dir / "models"),
            )
        return self._model

    def _cache_path(self, path: Path) -> Path:
        stat = path.stat()
        key = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{self.model_size}|ru|beam3"
        return self.cache_dir / (hashlib.sha256(key.encode("utf-8")).hexdigest() + ".json")

    def transcribe_file(self, path: str | Path) -> Transcription:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        cache_path = self._cache_path(path)
        with self._lock:
            if cache_path.exists():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                segments = tuple(ASRSegment(**item) for item in payload["segments"])
                return Transcription(text=str(payload["text"]), segments=segments)
            model = self._ensure_model()
            raw_segments, _ = model.transcribe(
                str(path), language="ru", beam_size=3,
                condition_on_previous_text=False, vad_filter=False,
            )
            segments = tuple(
                ASRSegment(float(item.start), float(item.end), str(item.text).strip())
                for item in raw_segments if str(item.text).strip()
            )
            result = Transcription(
                text=" ".join(item.text for item in segments).strip(),
                segments=segments,
            )
            payload = {
                "text": result.text,
                "segments": [item.__dict__ for item in segments],
            }
            cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return result
