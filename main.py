from __future__ import annotations

import sys

from bootstrap import bootstrap_and_relaunch





bootstrap_and_relaunch(include_build=False, entry_script="main.py")

from app.dependencies import ensure_dependencies

ensure_dependencies(auto_install=True)

from PySide6.QtWidgets import QApplication

from app.gui import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
