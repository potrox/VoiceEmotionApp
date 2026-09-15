"""Сборка Windows-приложения с помощью PyInstaller."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from bootstrap import bootstrap_and_relaunch

bootstrap_and_relaunch(include_build=True, entry_script="build.py")

ROOT = Path(__file__).resolve().parent
DIST_DIR = ROOT / "dist"
BUILD_DIR = ROOT / "build"
SPEC_FILE = ROOT / "VoiceEmotionApp.spec"


def main() -> int:
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    if SPEC_FILE.exists():
        SPEC_FILE.unlink()

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name",
        "VoiceEmotionApp",
        str(ROOT / "main.py"),
    ]

    print("Сборка VoiceEmotionApp с помощью PyInstaller...")
    subprocess.check_call(cmd, cwd=str(ROOT))
    print(f"Готово. Каталог приложения: {DIST_DIR / 'VoiceEmotionApp'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
