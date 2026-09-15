"""Точка входа настольного приложения VoiceEmotionApp."""
from __future__ import annotations

import sys

from bootstrap import bootstrap_and_relaunch


def main() -> int:
    bootstrap_and_relaunch(include_build=False, entry_script="main.py")

    from PySide6.QtWidgets import QApplication
    from app.ui.main_window import MainWindow

    application = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
