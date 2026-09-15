"""Загрузка, подготовка, проверка и воспроизведение аудиосигналов."""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

try:
    import librosa
except Exception as exc:
    librosa = None
    logger.exception("librosa is not available: %s", exc)

try:
    from scipy.signal import wiener
except Exception:
    wiener = None


@dataclass(frozen=True)
class AudioQuality:

    ok: bool
    warnings: list[str]
    duration_sec: float
    rms: float
    peak: float


def load_audio(
    path: str | Path, sample_rate: int = 16_000, mono: bool = True
) -> tuple[np.ndarray, int]:
    if librosa is None:
        raise RuntimeError(
            "Библиотека librosa не установлена. Установите зависимости из requirements.txt."
        )
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")
    try:
        audio_signal, actual_sample_rate = librosa.load(
            str(audio_path), sr=sample_rate, mono=mono
        )
    except Exception as exc:
        if audio_path.suffix.lower() == ".mp3":
            raise RuntimeError(
                f"Не удалось прочитать MP3-файл {audio_path}: {exc}. "
                "MP3 поддерживается при наличии совместимого аудиобэкенда "
                "soundfile/audioread. Для гарантированной обработки используйте "
                "WAV 16 кГц mono."
            ) from exc
        raise RuntimeError(
            f"Не удалось прочитать аудиофайл {audio_path}: {exc}"
        ) from exc
    if audio_signal.size == 0:
        raise RuntimeError(f"Аудиофайл пустой: {audio_path}")
    return audio_signal.astype(np.float32), int(actual_sample_rate)


def get_audio_duration(path: str | Path, sample_rate: int = 16_000) -> float:
    audio_path = Path(path)
    try:
        audio_info = sf.info(str(audio_path))
        if audio_info.frames and audio_info.samplerate:
            return float(audio_info.frames) / float(audio_info.samplerate)
    except Exception:
        pass
    audio_signal, actual_sample_rate = load_audio(
        audio_path, sample_rate=sample_rate, mono=True
    )
    return (
        float(len(audio_signal) / actual_sample_rate)
        if actual_sample_rate
        else 0.0
    )


def normalize_audio(audio_signal: np.ndarray) -> np.ndarray:
    normalized_signal = np.asarray(audio_signal, dtype=np.float32)
    peak_amplitude = (
        float(np.max(np.abs(normalized_signal))) if normalized_signal.size else 0.0
    )
    if peak_amplitude <= 1e-8:
        return normalized_signal
    return (normalized_signal / peak_amplitude * 0.95).astype(np.float32)


def trim_silence(audio_signal: np.ndarray, top_db: int = 28) -> np.ndarray:
    if librosa is None or audio_signal.size == 0:
        return audio_signal
    try:
        trimmed_signal, _ = librosa.effects.trim(audio_signal, top_db=top_db)
        if trimmed_signal.size > 0:
            return trimmed_signal.astype(np.float32)
    except Exception:
        logger.exception("Silence trimming failed")
    return audio_signal


def reduce_noise(audio_signal: np.ndarray) -> np.ndarray:
    if audio_signal.size == 0 or wiener is None:
        return audio_signal
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            filtered_signal = wiener(audio_signal, mysize=15)
        filtered_signal = np.nan_to_num(
            filtered_signal, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
        return (0.75 * filtered_signal + 0.25 * audio_signal).astype(np.float32)
    except Exception:
        logger.exception("Noise reduction failed")
        return audio_signal


def preprocess_signal(
    audio_signal: np.ndarray,
    *,
    normalize: bool = True,
    denoise: bool = True,
    trim: bool = True,
) -> np.ndarray:
    processed_signal = np.nan_to_num(np.asarray(audio_signal, dtype=np.float32))
    if trim:
        processed_signal = trim_silence(processed_signal)
    if denoise:
        processed_signal = reduce_noise(processed_signal)
    if normalize:
        processed_signal = normalize_audio(processed_signal)
    return processed_signal.astype(np.float32)


def preprocess_file(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 16_000,
    normalize: bool = True,
    denoise: bool = True,
    trim: bool = True,
) -> AudioQuality:
    audio_signal, _ = load_audio(input_path, sample_rate=sample_rate, mono=True)
    processed_signal = preprocess_signal(
        audio_signal, normalize=normalize, denoise=denoise, trim=trim
    )
    prepared_path = Path(output_path)
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(prepared_path), processed_signal, sample_rate, subtype="PCM_16")
    return analyze_quality(processed_signal, sample_rate, training=True)


def analyze_quality(
    audio_signal: np.ndarray,
    sample_rate: int,
    training: bool = False,
    min_duration_sec: float = 3.0,
) -> AudioQuality:
    quality_warnings: list[str] = []
    duration_seconds = (
        float(len(audio_signal) / sample_rate) if sample_rate else 0.0
    )
    peak_amplitude = (
        float(np.max(np.abs(audio_signal))) if audio_signal.size else 0.0
    )
    rms_amplitude = (
        float(np.sqrt(np.mean(np.square(audio_signal))))
        if audio_signal.size
        else 0.0
    )
    if min_duration_sec and duration_seconds < float(min_duration_sec):
        quality_warnings.append(
            f"Запись короче {float(min_duration_sec):.1f} секунд. "
            "Для обучения такая запись отклоняется; при распознавании надёжность снижена."
        )
    if peak_amplitude < 0.02 or rms_amplitude < 0.005:
        quality_warnings.append("Запись слишком тихая или почти пустая.")
    if rms_amplitude > 0 and peak_amplitude / max(rms_amplitude, 1e-8) < 2.0:
        quality_warnings.append("Запись может содержать высокий уровень постоянного шума.")

    critical_warnings = [
        warning
        for warning in quality_warnings
        if warning.startswith("Запись короче")
        or warning.startswith("Запись слишком тихая")
    ]
    return AudioQuality(
        ok=not critical_warnings if training else True,
        warnings=quality_warnings,
        duration_sec=duration_seconds,
        rms=rms_amplitude,
        peak=peak_amplitude,
    )


def split_long_signal(
    audio_signal: np.ndarray,
    sample_rate: int,
    max_duration_sec: float = 10.0,
) -> list[np.ndarray]:
    maximum_frames = int(max_duration_sec * sample_rate)
    if maximum_frames <= 0 or len(audio_signal) <= maximum_frames:
        return [audio_signal]
    chunks = [
        audio_signal[start_frame : start_frame + maximum_frames]
        for start_frame in range(0, len(audio_signal), maximum_frames)
        if len(audio_signal[start_frame : start_frame + maximum_frames]) >= sample_rate
    ]
    return chunks or [audio_signal]


def list_input_devices() -> list[dict[str, Any]]:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        try:
            default_input = sd.default.device[0]
        except Exception:
            default_input = None
    except Exception as exc:
        logger.warning("Input devices are unavailable: %s", exc)
        return []

    input_devices: list[dict[str, Any]] = []
    for device_index, device_info in enumerate(devices):
        try:
            input_channel_count = int(device_info.get("max_input_channels", 0))
        except Exception:
            input_channel_count = 0
        if input_channel_count <= 0:
            continue
        try:
            default_sample_rate = int(
                float(device_info.get("default_samplerate", 0)) or 0
            )
        except Exception:
            default_sample_rate = 0
        input_devices.append(
            {
                "index": device_index,
                "name": str(
                    device_info.get("name", f"Устройство {device_index}")
                ),
                "channels": input_channel_count,
                "default_samplerate": default_sample_rate,
                "is_default": device_index == default_input,
            }
        )
    return input_devices


def resample_signal(
    audio_signal: np.ndarray,
    actual_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    if int(actual_sample_rate) == int(target_sample_rate):
        return audio_signal.astype(np.float32)
    if librosa is None:
        raise RuntimeError(
            f"Микрофон записал звук с частотой {actual_sample_rate} Гц, "
            f"а рабочая частота приложения — {target_sample_rate} Гц. "
            "Для автоматического ресемплинга установите зависимости из requirements.txt."
        )
    return librosa.resample(
        audio_signal.astype(np.float32),
        orig_sr=int(actual_sample_rate),
        target_sr=int(target_sample_rate),
    ).astype(np.float32)


def save_wav(
    path: str | Path, audio_signal: np.ndarray, sample_rate: int = 16_000
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(output_path),
        np.asarray(audio_signal, dtype=np.float32),
        sample_rate,
        subtype="PCM_16",
    )
    return output_path


def play_signal(
    audio_signal: np.ndarray, sample_rate: int = 16_000, wait: bool = True
) -> None:
    try:
        import sounddevice as sd
    except Exception as exc:
        raise RuntimeError(
            "Библиотека sounddevice не установлена или устройство воспроизведения недоступно."
        ) from exc
    sd.stop()
    sd.play(np.asarray(audio_signal, dtype=np.float32), samplerate=sample_rate)
    if wait:
        sd.wait()


def play_wav_file(path: str | Path, wait: bool = False) -> None:
    audio_path = Path(path)
    if not audio_path.exists():
        raise RuntimeError(f"Файл записи не найден: {audio_path}")

    if sys.platform.startswith("win"):
        try:
            import winsound

            flags = winsound.SND_FILENAME
            if not wait:
                flags |= winsound.SND_ASYNC
            winsound.PlaySound(str(audio_path), flags)
            return
        except Exception as exc:
            raise RuntimeError(
                "Не удалось воспроизвести WAV-файл через системное устройство Windows. "
                "Проверьте громкость, выбранное устройство вывода и доступность файла."
            ) from exc

    try:
        audio_signal, sample_rate = sf.read(
            str(audio_path), dtype="float32", always_2d=False
        )
        if getattr(audio_signal, "ndim", 1) > 1:
            audio_signal = np.mean(audio_signal, axis=1)
        play_signal(
            np.asarray(audio_signal, dtype=np.float32),
            int(sample_rate),
            wait=wait,
        )
    except Exception as exc:
        raise RuntimeError(
            "Не удалось воспроизвести временный WAV-файл записи."
        ) from exc


def stop_playback() -> None:
    if sys.platform.startswith("win"):
        try:
            import winsound

            winsound.PlaySound(None, 0)
            return
        except Exception:
            pass
    try:
        import sounddevice as sd

        sd.stop()
    except Exception:
        pass
