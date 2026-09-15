"""Подготовка виртуального окружения и runtime-зависимостей."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
APP_VENV_ENV = "VOICEEMOTIONAPP_VENV"
DISABLE_VENV_ENV = "VOICEEMOTIONAPP_DISABLE_VENV"

RUNTIME_IMPORTS = [
    "PySide6",
    "numpy",
    "pandas",
    "librosa",
    "scipy",
    "sklearn",
    "matplotlib",
    "sounddevice",
    "soundfile",
    "audioread",
    "joblib",
    "torch",
    "requests",
]
BUILD_IMPORTS = ["PyInstaller"]
BUILD_REQUIREMENT = "pyinstaller>=6.0"


def running_in_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _python_tag() -> str:
    return f"py{sys.version_info.major}{sys.version_info.minor}"


def venv_dir() -> Path:
    custom = os.environ.get(APP_VENV_ENV, "").strip()
    if custom:
        return Path(custom).expanduser().resolve()

    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "VoiceEmotionApp" / f"venv-{_python_tag()}"

    return PROJECT_ROOT / ".venv"


def venv_python() -> Path:
    root = venv_dir()
    if os.name == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("[VoiceEmotionApp] " + " ".join(str(part) for part in cmd))
    subprocess.check_call(cmd, cwd=str(PROJECT_ROOT), env=env)


def _runtime_root() -> Path:
    custom = os.environ.get("VOICEEMOTIONAPP_RUNTIME", "").strip()
    if custom:
        return Path(custom).expanduser().resolve()



    return PROJECT_ROOT / ".runtime"


def _safe_env() -> dict[str, str]:
    env = os.environ.copy()
    runtime = _runtime_root()
    temp_dir = runtime / "tmp"
    cache_dir = runtime / "pip-cache"
    joblib_dir = runtime / "joblib"
    mpl_dir = runtime / "matplotlib"
    numba_dir = runtime / "numba"
    for path in [temp_dir, cache_dir, joblib_dir, mpl_dir, numba_dir]:
        path.mkdir(parents=True, exist_ok=True)



    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"




    env["TEMP"] = str(temp_dir)
    env["TMP"] = str(temp_dir)
    env["PIP_CACHE_DIR"] = str(cache_dir)
    env["JOBLIB_TEMP_FOLDER"] = str(joblib_dir)
    env["MPLCONFIGDIR"] = str(mpl_dir)
    env["NUMBA_CACHE_DIR"] = str(numba_dir)
    env.setdefault("VOICE_EMOTION_SKLEARN_N_JOBS", "1")
    return env


def _print_windows_path_advice() -> None:
    if os.name != "nt":
        return
    print()
    print("Windows note:")
    print("  PySide6 has deeply nested files. This installer uses a short venv path:")
    print(f"  {venv_dir()}")
    print("  If installation still fails with Long Path support, extract the project to:")
    print("  C:\\VoiceEmotionApp")
    print()


def ensure_local_venv() -> Path:
    py = venv_python()
    if py.exists():
        return py

    target = venv_dir()
    target.parent.mkdir(parents=True, exist_ok=True)
    _print_windows_path_advice()
    print(f"[VoiceEmotionApp] Creating virtual environment: {target}")
    _run([sys.executable, "-m", "venv", str(target)], env=_safe_env())

    if not py.exists():
        raise RuntimeError("Virtual environment was created, but python executable was not found.")
    return py


def dependencies_available(py: Path, include_build: bool = False) -> bool:
    imports = RUNTIME_IMPORTS + (BUILD_IMPORTS if include_build else [])
    code = (
        "import importlib.util, sys; "
        f"mods={imports!r}; "
        "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
        "print('Missing in venv: ' + ', '.join(missing) if missing else 'Dependencies are already installed in venv'); "
        "sys.exit(1 if missing else 0)"
    )
    imports_available = subprocess.call(
        [str(py), "-c", code], cwd=str(PROJECT_ROOT), env=_safe_env()
    ) == 0
    if not imports_available:
        return False
    dependency_marker = _dependency_marker(py, include_build)
    if not dependency_marker.exists():
        return False
    return dependency_marker.read_text(encoding="utf-8").strip() == (
        _requirements_fingerprint(include_build)
    )


def _requirements_fingerprint(include_build: bool) -> str:
    requirements_data = (PROJECT_ROOT / "requirements.txt").read_bytes()
    if include_build:
        requirements_data += BUILD_REQUIREMENT.encode("utf-8")
    return hashlib.sha256(requirements_data).hexdigest()


def _dependency_marker(py: Path, include_build: bool) -> Path:
    marker_name = ".build-requirements.sha256" if include_build else ".runtime-requirements.sha256"
    return py.parent.parent / marker_name


def install_dependencies_into_venv(py: Path, include_build: bool = False) -> None:
    req = PROJECT_ROOT / "requirements.txt"
    if not req.exists():
        raise FileNotFoundError(f"requirements.txt was not found: {req}")

    print("[VoiceEmotionApp] Installing dependencies into the virtual environment")
    _run([
        str(py), "-m", "pip", "install",
        "--disable-pip-version-check",
        "--no-warn-script-location",
        "--upgrade", "pip", "setuptools", "wheel",
    ], env=_safe_env())
    _run([
        str(py), "-m", "pip", "install",
        "--disable-pip-version-check",
        "--no-warn-script-location",
        "-r", str(req),
    ], env=_safe_env())

    if include_build:
        _run([
            str(py), "-m", "pip", "install",
            "--disable-pip-version-check",
            "--no-warn-script-location",
            BUILD_REQUIREMENT,
        ], env=_safe_env())
    _dependency_marker(py, False).write_text(
        _requirements_fingerprint(False), encoding="utf-8"
    )
    if include_build:
        _dependency_marker(py, True).write_text(
            _requirements_fingerprint(True), encoding="utf-8"
        )


def _entry_script_path(entry_script: str | Path | None) -> Path:
    if entry_script is None:
        raw = Path(sys.argv[0])
    else:
        raw = Path(entry_script)
    if raw.is_absolute():
        return raw
    return PROJECT_ROOT / raw


def bootstrap_and_relaunch(include_build: bool = False, entry_script: str | Path | None = None) -> None:
    if getattr(sys, "frozen", False):
        return
    if os.environ.get(DISABLE_VENV_ENV, "").strip() == "1":
        return
    if running_in_venv():
        current_python = Path(sys.executable)
        if not dependencies_available(current_python, include_build=include_build):
            install_dependencies_into_venv(
                current_python, include_build=include_build
            )
        return

    py = ensure_local_venv()
    if not dependencies_available(py, include_build=include_build):
        install_dependencies_into_venv(py, include_build=include_build)

    env = _safe_env()
    env["VOICEEMOTIONAPP_BOOTSTRAPPED"] = "1"
    script = _entry_script_path(entry_script)
    args = [str(py), str(script), *sys.argv[1:]]
    print("[VoiceEmotionApp] Перезапуск из виртуального окружения")
    raise SystemExit(subprocess.call(args, cwd=str(PROJECT_ROOT), env=env))
