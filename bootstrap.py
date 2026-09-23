from __future__ import annotations

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
    "sounddevice",
    "soundfile",
    "audioread",
    "joblib",
    "torch",
    "funasr",
    "faster_whisper",
    "requests",
]
BUILD_IMPORTS = ["PyInstaller"]


def running_in_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _python_tag() -> str:
    return f"py{sys.version_info.major}{sys.version_info.minor}"


def venv_dir() -> Path:
    """Return the virtual environment path used by the project.

    On Windows PySide6 contains very deeply nested QML files. Installing it into
    the Microsoft Store Python user-site path can exceed the classic MAX_PATH
    limit. Therefore the default Windows venv is placed into LocalAppData, where
    the path is much shorter and does not depend on the project folder depth.
    """
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
    numba_dir = runtime / "numba"
    for path in [temp_dir, cache_dir, joblib_dir, numba_dir]:
        path.mkdir(parents=True, exist_ok=True)



    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"




    env["TEMP"] = str(temp_dir)
    env["TMP"] = str(temp_dir)
    env["PIP_CACHE_DIR"] = str(cache_dir)
    env["JOBLIB_TEMP_FOLDER"] = str(joblib_dir)
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
        "import importlib.util, importlib.metadata, sys; "
        f"mods={imports!r}; "
        "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
        "version=importlib.metadata.version('scikit-learn') if 'sklearn' not in missing else 'missing'; "
        "missing+=['scikit-learn==1.8.0 (found '+version+')'] if version!='1.8.0' else []; "
        "print('Missing in venv: ' + ', '.join(missing) if missing else 'Dependencies are already installed in venv'); "
        "sys.exit(1 if missing else 0)"
    )
    return subprocess.call([str(py), "-c", code], cwd=str(PROJECT_ROOT), env=_safe_env()) == 0


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
            "pyinstaller",
        ], env=_safe_env())


def _entry_script_path(entry_script: str | Path | None) -> Path:
    if entry_script is None:
        raw = Path(sys.argv[0])
    else:
        raw = Path(entry_script)
    if raw.is_absolute():
        return raw
    return PROJECT_ROOT / raw


def bootstrap_and_relaunch(include_build: bool = False, entry_script: str | Path | None = None) -> None:
    """Ensure the command runs inside the VoiceEmotionApp virtual environment."""
    if sys.version_info < (3, 11):
        raise RuntimeError("VoiceEmotionApp v11 requires Python 3.11 or newer (scikit-learn 1.8.0).")
    if getattr(sys, "frozen", False):
        return
    if os.environ.get(DISABLE_VENV_ENV, "").strip() == "1":
        return
    if running_in_venv():
        return

    py = ensure_local_venv()
    if not dependencies_available(py, include_build=include_build):
        install_dependencies_into_venv(py, include_build=include_build)

    env = _safe_env()
    env["VOICEEMOTIONAPP_BOOTSTRAPPED"] = "1"
    script = _entry_script_path(entry_script)
    args = [str(py), str(script), *sys.argv[1:]]
    print("[VoiceEmotionApp] Restarting from the virtual environment")
    raise SystemExit(subprocess.call(args, cwd=str(PROJECT_ROOT), env=env))
