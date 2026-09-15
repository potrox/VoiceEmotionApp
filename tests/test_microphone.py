"""Проверки записи и освобождения аудиопотока."""

from __future__ import annotations

import numpy as np

from app.microphone import MicrophoneRecorder


class FakeInputStream:
    def __init__(self, callback, audio_block: np.ndarray) -> None:
        self.callback = callback
        self.audio_block = audio_block
        self.aborted = False
        self.closed = False

    def start(self) -> None:
        self.callback(
            self.audio_block.reshape(-1, 1),
            len(self.audio_block),
            None,
            "input overflow",
        )

    def abort(self) -> None:
        self.aborted = True

    def close(self) -> None:
        self.closed = True


class FakeAudioBackend:
    def __init__(self, audio_block: np.ndarray) -> None:
        self.audio_block = audio_block
        self.stream: FakeInputStream | None = None

    def check_input_settings(self, **_settings) -> None:
        return None

    def InputStream(self, callback, **_settings) -> FakeInputStream:
        self.stream = FakeInputStream(callback, self.audio_block)
        return self.stream


def test_recorder_limits_fixed_duration_and_closes_stream() -> None:
    backend = FakeAudioBackend(np.arange(20, dtype=np.float32))
    recorder = MicrophoneRecorder(target_sample_rate=1_000, audio_backend=backend)

    recorder.start(duration_seconds=0.01)
    recording = recorder.stop()

    assert recording.audio_signal.tolist() == list(np.arange(10, dtype=np.float32))
    assert recording.duration_seconds == 0.01
    assert recording.driver_warnings == ("input overflow",)
    assert recorder.is_active is False
    assert backend.stream is not None
    assert backend.stream.aborted is True
    assert backend.stream.closed is True


def test_cancel_discards_captured_audio() -> None:
    backend = FakeAudioBackend(np.ones(8, dtype=np.float32))
    recorder = MicrophoneRecorder(target_sample_rate=8_000, audio_backend=backend)

    recorder.start()
    recorder.cancel()

    assert recorder.is_active is False
    assert recorder.progress is None
