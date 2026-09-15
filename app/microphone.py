"""Единый сервис записи с микрофона, независимый от интерфейса Qt."""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .audio import resample_signal


logger = logging.getLogger(__name__)


class MicrophoneError(RuntimeError):
    pass


class EmptyRecordingError(MicrophoneError):
    pass


@dataclass(frozen=True)
class AudioRecording:

    audio_signal: np.ndarray
    sample_rate: int
    driver_warnings: tuple[str, ...]

    @property
    def duration_seconds(self) -> float:
        return float(len(self.audio_signal) / self.sample_rate) if self.sample_rate else 0.0


class MicrophoneRecorder:

    def __init__(
        self,
        target_sample_rate: int,
        device_index: int | None = None,
        audio_backend: Any | None = None,
    ) -> None:
        self.target_sample_rate = int(target_sample_rate)
        self.device_index = device_index
        self._audio_backend = audio_backend
        self._stream: Any | None = None
        self._audio_blocks: list[np.ndarray] = []
        self._driver_warnings: list[str] = []
        self._expected_frames: int | None = None
        self._received_frames = 0
        self._actual_sample_rate = self.target_sample_rate
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def actual_sample_rate(self) -> int:
        return self._actual_sample_rate

    @property
    def progress(self) -> float | None:
        if self._expected_frames is None:
            return None
        return min(self._received_frames / max(self._expected_frames, 1), 1.0)

    def start(self, duration_seconds: float | None = None) -> int:
        if self._active:
            raise MicrophoneError("Запись с этого микрофона уже запущена.")
        if duration_seconds is not None and duration_seconds <= 0:
            raise ValueError("Длительность записи должна быть больше нуля.")

        audio_backend = self._load_audio_backend()
        device = self._normalized_device_index()
        sample_rate = self._select_supported_sample_rate(audio_backend, device)

        self._audio_blocks.clear()
        self._driver_warnings.clear()
        self._received_frames = 0
        self._actual_sample_rate = sample_rate
        self._expected_frames = (
            max(1, int(duration_seconds * sample_rate))
            if duration_seconds is not None
            else None
        )

        try:
            self._stream = audio_backend.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=device,
                callback=self._capture_audio_block,
            )
            self._active = True
            self._stream.start()
        except Exception as exc:
            self._active = False
            self._stream = None
            raise MicrophoneError(
                "Не удалось запустить поток микрофона. Проверьте, не занят ли он другой программой."
            ) from exc
        return sample_rate

    def stop(self, pad_to_requested_duration: bool = False) -> AudioRecording:
        self._active = False
        self._close_stream()
        if not self._audio_blocks:
            self._reset_buffers()
            raise EmptyRecordingError("Микрофон не передал аудиоданные.")

        audio_signal = np.concatenate(self._audio_blocks).astype(np.float32)
        if self._expected_frames is not None:
            if audio_signal.size > self._expected_frames:
                audio_signal = audio_signal[: self._expected_frames]
            elif pad_to_requested_duration and audio_signal.size < self._expected_frames:
                audio_signal = np.pad(
                    audio_signal,
                    (0, self._expected_frames - audio_signal.size),
                    mode="constant",
                )

        audio_signal = resample_signal(
            audio_signal, self._actual_sample_rate, self.target_sample_rate
        )
        recording = AudioRecording(
            audio_signal=audio_signal,
            sample_rate=self.target_sample_rate,
            driver_warnings=tuple(self._driver_warnings),
        )
        self._reset_buffers()
        return recording

    def record(self, duration_seconds: float) -> AudioRecording:
        audio_backend = self._load_audio_backend()
        self.start(duration_seconds)
        deadline = time.monotonic() + float(duration_seconds)
        try:
            while time.monotonic() < deadline:
                if hasattr(audio_backend, "sleep"):
                    audio_backend.sleep(25)
                else:
                    time.sleep(0.025)
        except Exception:
            self.cancel()
            raise
        return self.stop(pad_to_requested_duration=True)

    def cancel(self) -> None:
        self._active = False
        self._close_stream()
        self._reset_buffers()

    def _load_audio_backend(self) -> Any:
        if self._audio_backend is not None:
            return self._audio_backend
        try:
            self._audio_backend = importlib.import_module("sounddevice")
        except (ImportError, OSError) as exc:
            raise MicrophoneError(
                "Библиотека sounddevice недоступна или не может открыть аудиоустройства."
            ) from exc
        return self._audio_backend

    def _normalized_device_index(self) -> int | None:
        if self.device_index is None or int(self.device_index) < 0:
            return None
        return int(self.device_index)

    def _select_supported_sample_rate(self, audio_backend: Any, device: int | None) -> int:
        try:
            audio_backend.check_input_settings(
                device=device, samplerate=self.target_sample_rate, channels=1
            )
            return self.target_sample_rate
        except Exception:
            try:
                device_info = (
                    audio_backend.query_devices(device, "input")
                    if device is not None
                    else audio_backend.query_devices(kind="input")
                )
                sample_rate = int(
                    float(device_info.get("default_samplerate", self.target_sample_rate))
                    or self.target_sample_rate
                )
                audio_backend.check_input_settings(
                    device=device, samplerate=sample_rate, channels=1
                )
                return sample_rate
            except Exception as exc:
                raise MicrophoneError(
                    "Не удалось открыть выбранный микрофон. Проверьте разрешения Windows и устройство записи."
                ) from exc

    def _capture_audio_block(
        self, input_audio: np.ndarray, frame_count: int, _time_info: Any, status: Any
    ) -> None:
        if status:
            warning = str(status)
            if warning and warning not in self._driver_warnings:
                self._driver_warnings.append(warning)
        if not self._active or frame_count <= 0:
            return

        frames_to_take = int(frame_count)
        if self._expected_frames is not None:
            remaining_frames = self._expected_frames - self._received_frames
            if remaining_frames <= 0:
                return
            frames_to_take = min(frames_to_take, remaining_frames)

        audio_block = np.array(
            input_audio[:frames_to_take, 0], dtype=np.float32, copy=True
        )
        self._audio_blocks.append(audio_block)
        self._received_frames += frames_to_take

    def _close_stream(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try:
            stream.abort()
        except Exception:
            try:
                stream.stop()
            except Exception as exc:
                logger.debug("Не удалось остановить поток микрофона", exc_info=exc)
        try:
            stream.close()
        except Exception as exc:
            logger.debug("Не удалось закрыть поток микрофона", exc_info=exc)

    def _reset_buffers(self) -> None:
        self._audio_blocks.clear()
        self._driver_warnings.clear()
        self._expected_frames = None
        self._received_frames = 0
