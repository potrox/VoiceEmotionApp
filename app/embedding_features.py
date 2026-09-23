from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import threading
from pathlib import Path

import numpy as np

from .audio import save_wav


logger = logging.getLogger(__name__)


class Emotion2VecExtractor:
    """Lazy CPU extractor for frozen utterance-level emotion2vec embeddings."""

    def __init__(self, model_id: str, dimension: int = 768, ncpu: int = 4):
        self.model_id = str(model_id)
        self.dimension = int(dimension)
        self.ncpu = max(1, int(ncpu))
        self._model = None
        self._lock = threading.RLock()

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from funasr import AutoModel
            except Exception as exc:
                raise RuntimeError(
                    "Для этой модели нужен пакет funasr. Запустите install.bat "
                    "или установите зависимости из requirements.txt."
                ) from exc
            logging.disable(logging.INFO)
            try:
                with open(os.devnull, "w", encoding="utf-8") as sink:
                    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                        self._model = AutoModel(
                            model=self.model_id,
                            hub="hf",
                            device="cpu",
                            disable_update=True,
                            disable_pbar=True,
                            ncpu=self.ncpu,
                        )
            finally:
                logging.disable(logging.NOTSET)
            return self._model

    def extract_file(self, path: str | Path) -> np.ndarray:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        model = self._ensure_model()
        with self._lock:
            with open(os.devnull, "w", encoding="utf-8") as sink:
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    generated = model.generate(
                        input=[str(path)],
                        granularity="utterance",
                        extract_embedding=True,
                        batch_size=1,
                    )
        if not generated:
            raise RuntimeError(f"emotion2vec не вернул эмбеддинг для файла: {path}")
        vector = np.asarray(generated[0]["feats"], dtype=np.float32).reshape(-1)
        if vector.shape != (self.dimension,):
            raise RuntimeError(
                f"Неожиданный размер emotion2vec: {vector.shape}; ожидался {self.dimension}."
            )
        return vector

    def extract_signal(self, y: np.ndarray, sr: int) -> np.ndarray:
        handle, temp_name = tempfile.mkstemp(prefix="voice_emotion_", suffix=".wav")
        os.close(handle)
        temp_path = Path(temp_name)
        try:
            save_wav(temp_path, np.asarray(y, dtype=np.float32), int(sr))
            return self.extract_file(temp_path)
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                logger.warning("Could not remove temporary audio file: %s", temp_path)
