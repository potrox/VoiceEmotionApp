"""Установка приложения и создание рабочих каталогов."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from bootstrap import (
    bootstrap_and_relaunch,
    dependencies_available,
    install_dependencies_into_venv,
    venv_dir,
)

bootstrap_and_relaunch(include_build=False, entry_script="install.py")


def main() -> int:
    parser = argparse.ArgumentParser(description="Установка зависимостей VoiceEmotionApp")
    parser.add_argument(
        "--build",
        action="store_true",
        help="также установить зависимости для сборки EXE",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="принудительно выполнить pip install -r requirements.txt",
    )
    args = parser.parse_args()

    print("Установка зависимостей VoiceEmotionApp")
    print(f"Python: {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"Виртуальное окружение: {venv_dir()}")
    if sys.version_info < (3, 11):
        print("ПРЕДУПРЕЖДЕНИЕ: требуется Python 3.11 или новее.")
    if sys.version_info >= (3, 13):
        print(
            "ПРЕДУПРЕЖДЕНИЕ: для используемых ML- и аудиобиблиотек "
            "рекомендуется Python 3.11 или 3.12."
        )

    try:
        current_python = Path(sys.executable)
        if args.force or not dependencies_available(
            current_python, include_build=args.build
        ):
            install_dependencies_into_venv(
                current_python, include_build=args.build
            )
    except Exception as exc:
        print()
        print("ОШИБКА УСТАНОВКИ:")
        print(exc)
        return 1

    print("Зависимости установлены и доступны.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
