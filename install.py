from __future__ import annotations

import argparse
import platform
import sys

from bootstrap import bootstrap_and_relaunch, venv_dir

bootstrap_and_relaunch(include_build=False, entry_script="install.py")

from app.dependencies import ensure_dependencies, find_missing_dependencies, run_pip_install


def main() -> int:
    parser = argparse.ArgumentParser(description="Install VoiceEmotionApp dependencies")
    parser.add_argument("--build", action="store_true", help="also check dependencies for exe build")
    parser.add_argument("--force", action="store_true", help="force pip install -r requirements.txt")
    args = parser.parse_args()

    print("VoiceEmotionApp dependency installer")
    print(f"Python: {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"Virtual environment: {venv_dir()}")
    if sys.version_info < (3, 11):
        print("WARNING: Python 3.11 is recommended for the saved v11 model.")
    if sys.version_info >= (3, 13):
        print("WARNING: Python 3.11 is recommended for the saved v11 model and audio packages.")

    try:
        if args.force:
            run_pip_install()
        ensure_dependencies(include_build=args.build, auto_install=True)
    except Exception as exc:
        print()
        print("INSTALLATION ERROR:")
        print(exc)
        return 1

    missing = find_missing_dependencies(include_build=args.build)
    if missing:
        print("Could not confirm that all dependencies are installed.")
        return 1

    print("Done: dependencies are installed and available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
