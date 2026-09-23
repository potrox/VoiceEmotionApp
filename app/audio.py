from __future__ import annotations

import logging
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


@dataclass
class AudioQuality:
    ok: bool
    warnings: List[str]
    duration_sec: float
    rms: float
    peak: float


def load_audio(path: str | Path, sample_rate: int = 16000, mono: bool = True) -> Tuple[np.ndarray, int]:
    if librosa is None:
        raise RuntimeError("Библиотека librosa не установлена. Установите зависимости из requirements.txt.")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Аудиофайл не найден: {path}")
    try:
        y, sr = librosa.load(str(path), sr=sample_rate, mono=mono)
    except Exception as exc:
        if path.suffix.lower() == ".mp3":
            raise RuntimeError(
                f"Не удалось прочитать MP3-файл {path}: {exc}. "
                "MP3 поддерживается при наличии совместимого аудиобэкенда soundfile/audioread. "
                "Для гарантированной обработки используйте WAV 16 кГц mono."
            ) from exc
        raise RuntimeError(f"Не удалось прочитать аудиофайл {path}: {exc}") from exc
    if y.size == 0:
        raise RuntimeError(f"Аудиофайл пустой: {path}")
    return y.astype(np.float32), int(sr)




def get_audio_duration(path: str | Path, sample_rate: int = 16000) -> float:
    """Возвращает длительность аудиофайла без изменения самого файла.

    Для WAV сначала используется soundfile.info, чтобы не зависеть от
    предобработки, удаления тишины или ресемплинга. Если быстрый способ
    недоступен, файл читается через load_audio как запасной вариант.
    """
    path = Path(path)
    try:
        info = sf.info(str(path))
        if info.frames and info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    y, sr = load_audio(path, sample_rate=sample_rate, mono=True)
    return float(len(y) / sr) if sr else 0.0

def normalize_audio(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 1e-8:
        return y
    return (y / peak * 0.95).astype(np.float32)


def trim_silence(y: np.ndarray, top_db: int = 28) -> np.ndarray:
    if librosa is None or y.size == 0:
        return y
    try:
        yt, _ = librosa.effects.trim(y, top_db=top_db)
        if yt.size > 0:
            return yt.astype(np.float32)
    except Exception:
        logger.exception("Silence trimming failed")
    return y


def reduce_noise(y: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return y
    if wiener is None:
        return y
    try:



        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            filtered = wiener(y, mysize=15)
        filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return (0.75 * filtered + 0.25 * y).astype(np.float32)
    except Exception:
        logger.exception("Noise reduction failed")
        return y

def preprocess_signal(
    y: np.ndarray,
    sr: int,
    normalize: bool = True,
    denoise: bool = True,
    trim: bool = True,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    y = np.nan_to_num(y)
    if trim:
        y = trim_silence(y)
    if denoise:
        y = reduce_noise(y)
    if normalize:
        y = normalize_audio(y)
    return y.astype(np.float32)


def preprocess_file(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 16000,
    normalize: bool = True,
    denoise: bool = True,
    trim: bool = True,
) -> AudioQuality:
    y, sr = load_audio(input_path, sample_rate=sample_rate, mono=True)
    y = preprocess_signal(y, sr, normalize=normalize, denoise=denoise, trim=trim)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), y, sample_rate, subtype="PCM_16")
    return analyze_quality(y, sample_rate, training=True)


def analyze_quality(y: np.ndarray, sr: int, training: bool = False, min_duration_sec: float = 3.0) -> AudioQuality:
    warnings_list: List[str] = []
    duration = float(len(y) / sr) if sr else 0.0
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0
    if min_duration_sec and duration < float(min_duration_sec):
        msg = (
            f"Запись короче {float(min_duration_sec):.1f} секунд. "
            "Для обучения такая запись отклоняется; при распознавании надёжность снижена."
        )
        warnings_list.append(msg)
    if peak < 0.02 or rms < 0.005:
        warnings_list.append("Запись слишком тихая или почти пустая.")
    if rms > 0 and peak / max(rms, 1e-8) < 2.0:
        warnings_list.append("Запись может содержать высокий уровень постоянного шума.")
    if training:




        critical = [
            msg for msg in warnings_list
            if msg.startswith("Запись короче") or msg.startswith("Запись слишком тихая")
        ]
        ok = not critical
    else:
        ok = True
    return AudioQuality(ok=ok, warnings=warnings_list, duration_sec=duration, rms=rms, peak=peak)

def split_long_signal(y: np.ndarray, sr: int, max_duration_sec: float = 10.0) -> List[np.ndarray]:
    max_len = int(max_duration_sec * sr)
    if max_len <= 0 or len(y) <= max_len:
        return [y]
    chunks: List[np.ndarray] = []
    for start in range(0, len(y), max_len):
        chunk = y[start:start + max_len]
        if len(chunk) >= int(1.0 * sr):
            chunks.append(chunk)
    return chunks or [y]


def list_input_devices() -> List[Dict[str, Any]]:
    """Return available input devices for the GUI.

    sounddevice can fail on systems without PortAudio devices; in that case
    the GUI receives an empty list and shows a clear message instead of
    silently doing nothing.
    """
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        default_input = None
        try:
            default_input = sd.default.device[0]
        except Exception:
            default_input = None
    except Exception as exc:
        logger.warning("Input devices are unavailable: %s", exc)
        return []

    result: List[Dict[str, Any]] = []
    for index, dev in enumerate(devices):
        try:
            input_channels = int(dev.get("max_input_channels", 0))
        except Exception:
            input_channels = 0
        if input_channels <= 0:
            continue
        try:
            default_sr = int(float(dev.get("default_samplerate", 0)) or 0)
        except Exception:
            default_sr = 0
        result.append({
            "index": index,
            "name": str(dev.get("name", f"Устройство {index}")),
            "channels": input_channels,
            "default_samplerate": default_sr,
            "is_default": index == default_input,
        })
    return result


def _resample_if_needed(y: np.ndarray, actual_sr: int, target_sr: int) -> np.ndarray:
    if int(actual_sr) == int(target_sr):
        return y.astype(np.float32)
    if librosa is None:
        raise RuntimeError(
            f"Микрофон записал звук с частотой {actual_sr} Гц, а рабочая частота приложения — {target_sr} Гц. "
            "Для автоматического ресемплинга установите зависимости из requirements.txt."
        )
    return librosa.resample(y.astype(np.float32), orig_sr=int(actual_sr), target_sr=int(target_sr)).astype(np.float32)


def record_microphone(duration_sec: int, sample_rate: int = 16000, device_index: Optional[int] = None) -> np.ndarray:
    """Record audio from a selected microphone and return strictly after duration_sec.

    The function uses sounddevice.InputStream with a callback and closes the
    stream by abort(), not by sd.wait(). This avoids the common Windows problem
    where recording reaches 99% in the GUI and then waits forever for the audio
    driver to report natural completion.
    """
    try:
        import sounddevice as sd
    except Exception as exc:
        raise RuntimeError("Библиотека sounddevice не установлена или микрофон недоступен.") from exc

    duration_sec = int(duration_sec)
    if duration_sec <= 0:
        raise ValueError("Длительность записи должна быть положительной.")

    device = None if device_index is None or int(device_index) < 0 else int(device_index)
    if not list_input_devices():
        raise RuntimeError("Не найдено ни одного устройства записи. Подключите микрофон и нажмите «Обновить устройства».")

    actual_sr = int(sample_rate)
    device_label = f"устройство #{device}" if device is not None else "устройство по умолчанию"
    try:
        sd.check_input_settings(device=device, samplerate=actual_sr, channels=1)
    except Exception:
        try:
            dev_info = sd.query_devices(device, "input") if device is not None else sd.query_devices(kind="input")
            actual_sr = int(float(dev_info.get("default_samplerate", sample_rate)) or sample_rate)
            sd.check_input_settings(device=device, samplerate=actual_sr, channels=1)
        except Exception as exc:
            raise RuntimeError(
                f"Не удалось открыть микрофон ({device_label}). Выберите другое устройство записи или проверьте доступ приложения к микрофону. "
                f"Подробности: {exc}"
            ) from exc

    target_frames = max(1, int(duration_sec * actual_sr))
    received_frames = 0
    chunks: List[np.ndarray] = []
    statuses: List[str] = []

    def callback(indata, frames, _time_info, status):
        nonlocal received_frames
        if status:
            text = str(status)
            if text and text not in statuses:
                statuses.append(text)
        remaining = target_frames - received_frames
        if remaining <= 0:
            return
        take = min(int(frames), int(remaining))
        if take > 0:
            chunks.append(np.array(indata[:take, 0], dtype=np.float32, copy=True))
            received_frames += take

    stream = None
    try:
        stream = sd.InputStream(
            samplerate=actual_sr,
            channels=1,
            dtype="float32",
            device=device,
            callback=callback,
        )
        stream.start()
        deadline = time.monotonic() + float(duration_sec)
        while time.monotonic() < deadline:
            sd.sleep(25)
    except Exception as exc:
        raise RuntimeError(
            f"Ошибка записи с микрофона ({device_label}). Проверьте, что микрофон не занят другой программой и разрешён в Windows. "
            f"Подробности: {exc}"
        ) from exc
    finally:
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                try:
                    stream.stop()
                except Exception:
                    pass
            try:
                stream.close()
            except Exception:
                pass

    if not chunks:
        raise RuntimeError("Микрофон не вернул данные записи. Проверьте выбранное устройство записи и разрешения Windows.")

    y = np.concatenate(chunks).astype(np.float32)
    if y.size < target_frames:
        y = np.pad(y, (0, target_frames - y.size), mode="constant")
    elif y.size > target_frames:
        y = y[:target_frames]

    y = _resample_if_needed(y, actual_sr, sample_rate)
    return y.astype(np.float32)

def save_wav(path: str | Path, y: np.ndarray, sr: int = 16000) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(y, dtype=np.float32), sr, subtype="PCM_16")
    return path


def play_signal(y: np.ndarray, sr: int = 16000, wait: bool = True) -> None:
    try:
        import sounddevice as sd
    except Exception as exc:
        raise RuntimeError("Библиотека sounddevice не установлена или устройство воспроизведения недоступно.") from exc
    sd.stop()
    sd.play(np.asarray(y, dtype=np.float32), samplerate=sr)
    if wait:
        sd.wait()


def play_wav_file(path: str | Path, wait: bool = False) -> None:
    """Проигрывает WAV-файл.

    На Windows для прослушивания пользовательской записи используется winsound: он
    работает с обычными PCM WAV-файлами без выбора отдельного устройства вывода и
    не блокирует интерфейс в асинхронном режиме. На других системах используется
    soundfile + sounddevice как запасной вариант.
    """
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Файл записи не найден: {path}")

    if sys.platform.startswith("win"):
        try:
            import winsound
            flags = winsound.SND_FILENAME
            if not wait:
                flags |= winsound.SND_ASYNC
            winsound.PlaySound(str(path), flags)
            return
        except Exception as exc:
            raise RuntimeError(
                "Не удалось воспроизвести WAV-файл через системное устройство Windows. "
                "Проверьте громкость, выбранное устройство вывода и доступность файла."
            ) from exc

    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if getattr(y, "ndim", 1) > 1:
            y = np.mean(y, axis=1)
        play_signal(np.asarray(y, dtype=np.float32), int(sr), wait=wait)
    except Exception as exc:
        raise RuntimeError("Не удалось воспроизвести временный WAV-файл записи.") from exc


def stop_playback() -> None:
    """Останавливает текущее прослушивание, если оно было запущено."""
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
