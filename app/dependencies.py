from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class Dependency:
    import_name: str
    requirement_name: str
    purpose: str


RUNTIME_DEPENDENCIES: List[Dependency] = [
    Dependency("PySide6", "PySide6", "graphical interface"),
    Dependency("numpy", "numpy", "numeric calculations"),
    Dependency("pandas", "pandas", "CSV datasets and tables"),
    Dependency("librosa", "librosa", "audio processing and features"),
    Dependency("scipy", "scipy", "signal processing"),
    Dependency("sklearn", "scikit-learn", "Logistic Regression, Random Forest, MLP, metrics"),
    Dependency("sounddevice", "sounddevice", "microphone recording and playback"),
    Dependency("soundfile", "soundfile", "WAV reading and writing"),
    Dependency("audioread", "audioread", "additional audio reading, including MP3"),
    Dependency("joblib", "joblib", "caching and object serialization"),
    Dependency("torch", "torch", "emotion2vec inference runtime"),
    Dependency("funasr", "funasr", "emotion2vec feature extraction"),
    Dependency("faster_whisper", "faster-whisper", "local Russian speech transcription"),
]

BUILD_DEPENDENCIES: List[Dependency] = [
    Dependency("PyInstaller", "pyinstaller", "exe build"),
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def requirements_path() -> Path:
    return project_root() / "requirements.txt"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _is_module_available(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None


def find_missing_dependencies(include_build: bool = False) -> List[Dependency]:
    dependencies = list(RUNTIME_DEPENDENCIES)
    if include_build:
        dependencies.extend(BUILD_DEPENDENCIES)
    return [dep for dep in dependencies if not _is_module_available(dep.import_name)]


def _print_dependency_list(prefix: str, dependencies: Iterable[Dependency]) -> None:
    deps = list(dependencies)
    if not deps:
        return
    print(prefix)
    for dep in deps:
        print(f"  - {dep.requirement_name}: {dep.purpose}")


def _pip_env() -> dict[str, str]:
    env = os.environ.copy()
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            base = Path(local) / "VoiceEmotionApp"
            temp_dir = base / "tmp"
            cache_dir = base / "pip-cache"
            temp_dir.mkdir(parents=True, exist_ok=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            env["TEMP"] = str(temp_dir)
            env["TMP"] = str(temp_dir)
            env["PIP_CACHE_DIR"] = str(cache_dir)
    return env


def run_pip_install(requirements: Path | None = None) -> None:
    req = requirements or requirements_path()
    if not req.exists():
        raise FileNotFoundError(f"requirements.txt was not found: {req}")

    print("Updating pip, setuptools and wheel...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--disable-pip-version-check",
        "--no-warn-script-location",
        "--upgrade", "pip", "setuptools", "wheel",
    ], env=_pip_env())

    print(f"Installing libraries from {req.name}...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--disable-pip-version-check",
        "--no-warn-script-location",
        "-r", str(req),
    ], env=_pip_env())


def ensure_dependencies(include_build: bool = False, auto_install: bool = True) -> None:
    if is_frozen():
        return

    missing = find_missing_dependencies(include_build=include_build)
    if not missing:
        return

    _print_dependency_list("Missing required libraries:", missing)

    if os.environ.get("VOICEEMOTIONAPP_SKIP_AUTO_INSTALL", "").strip() == "1":
        names = ", ".join(dep.requirement_name for dep in missing)
        raise RuntimeError(
            "Automatic dependency installation is disabled by "
            f"VOICEEMOTIONAPP_SKIP_AUTO_INSTALL=1. Install manually: {names}"
        )

    if not auto_install:
        names = ", ".join(dep.requirement_name for dep in missing)
        raise RuntimeError(f"Missing dependencies: {names}")

    print("Automatic dependency installation will be started.")
    try:
        run_pip_install()
    except subprocess.CalledProcessError as exc:
        names = ", ".join(dep.requirement_name for dep in missing)
        raise RuntimeError(
            "Could not install dependencies automatically. "
            "Check internet connection, Python version and permissions. "
            "If the error mentions Windows Long Path support, extract the project "
            "to C:\\VoiceEmotionApp and run install.bat again. "
            f"Then run manually: {sys.executable} -m pip install -r requirements.txt. "
            f"Missing: {names}"
        ) from exc

    still_missing = find_missing_dependencies(include_build=include_build)
    if still_missing:
        names = ", ".join(dep.requirement_name for dep in still_missing)
        raise RuntimeError(
            "Installation finished, but some libraries are still unavailable: "
            f"{names}. Try running: {sys.executable} -m pip install -r requirements.txt"
        )

    print("All required libraries are installed.")
