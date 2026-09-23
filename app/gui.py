from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, QTimer, Slot
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QTabWidget,
    QTextEdit,
)
import numpy as np

from .audio import stop_playback
from .config import AppConfig, ProjectPaths
from .dataset import DatasetLoadResult
from .gui_dataset import DatasetTabMixin
from .gui_home import HomeTabMixin
from .gui_recognition import RecognitionTabMixin
from .gui_recording import RecordingTabMixin
from .gui_settings import SettingsTabMixin
from .gui_shared import DARK_QSS, LIGHT_QSS
from .gui_testing import TestingTabMixin
from .gui_training import TrainingTabMixin
from .recognizer import Recognizer
from .storage import setup_logging


class MainWindow(
    HomeTabMixin,
    DatasetTabMixin,
    RecordingTabMixin,
    TrainingTabMixin,
    TestingTabMixin,
    RecognitionTabMixin,
    SettingsTabMixin,
    QMainWindow,
):
    def __init__(self):
        super().__init__()
        self.cfg = AppConfig.load("config.json")
        self.paths = ProjectPaths(self.cfg)
        self.paths.ensure()
        setup_logging(self.paths)
        self.dataset_result: Optional[DatasetLoadResult] = None
        self.current_model_path: Optional[str] = None
        self._recognizer_cache: Optional[Recognizer] = None
        self._recognizer_cache_path: Optional[str] = None
        self.recording_buffer = None
        self.recording_cache_path: Optional[Path] = None
        self.is_playing_recording = False
        self.is_recording_now = False
        self.recording_generation = 0
        self.record_started_at = 0.0
        self.record_expected_duration = 0.0
        self.record_timer = QTimer(self)
        self.record_timer.setInterval(200)
        self.record_timer.timeout.connect(self._update_record_timer_progress)
        self.live_record_stream = None
        self.live_record_blocks: list[np.ndarray] = []
        self.live_record_statuses: list[str] = []
        self.live_record_actual_sr = self.cfg.sample_rate
        self.live_record_expected_frames = 0
        self.live_record_received_frames = 0
        self.is_recognition_mic_recording = False
        self.recognition_mic_stream = None
        self.recognition_mic_blocks: list[np.ndarray] = []
        self.recognition_mic_statuses: list[str] = []
        self.recognition_mic_actual_sr = self.cfg.sample_rate
        self.recognition_mic_started_at = 0.0
        self.recognition_mic_timer = QTimer(self)
        self.recognition_mic_timer.setInterval(250)
        self.recognition_mic_timer.timeout.connect(self._update_recognition_mic_timer)
        self.threads: list[QThread] = []


        self.workers: list[object] = []
        self.worker_contexts: dict[int, dict] = {}
        self.setWindowTitle("VoiceEmotionApp")
        self.resize(1320, 880)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self._build_home_tab()
        self._build_dataset_tab()
        self._build_record_tab()
        self._build_training_tab()
        self._build_testing_tab()
        self._build_recognition_tab()
        self._build_settings_tab()
        self._apply_theme(self.cfg.theme)
        self._load_best_model_pointer()
        self._refresh_home()
        self._refresh_history()
        self._refresh_record_progress()
        QTimer.singleShot(700, self._recover_user_dataset_csv_on_startup)

    def closeEvent(self, event) -> None:
        stop_playback()
        if self.is_recognition_mic_recording:
            self._cancel_recognition_mic_recording()
        if self.is_recording_now:
            self._cancel_live_recording()
        elif self.recording_cache_path and self.recording_buffer is not None:

            self._clear_recording_cache_file()
        super().closeEvent(event)

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def _show_info(self, title: str, message: str) -> None:
        QMessageBox.information(self, title, message)

    @Slot(int, str)
    def _on_worker_progress(self, value: int, text: str) -> None:
        """Update GUI from worker progress in the main Qt thread.

        Важно: нельзя обновлять QTextEdit/QProgressBar из фонового потока через
        lambda-слоты. На Windows это приводило к сообщениям вида
        QObject::setParent/QBasicTimer и пустому окну ошибки. Все обновления UI
        проходят через этот слот главного окна.
        """
        worker = self.sender()
        ctx = self.worker_contexts.get(id(worker), {}) if worker is not None else {}
        progress_bar = ctx.get("progress_bar")
        log = ctx.get("log")
        if progress_bar is not None:
            progress_bar.setValue(int(value))
        if log is not None:
            log.append(f"[{int(value)}%] {text}")

    @Slot(str)
    def _on_worker_failed(self, message: str) -> None:
        """Show worker errors safely in the GUI thread."""
        worker = self.sender()
        ctx = self.worker_contexts.get(id(worker), {}) if worker is not None else {}
        text = str(message).strip() or "Операция завершилась ошибкой, но подробное сообщение не было передано. Проверьте logs/."
        log = ctx.get("log")
        if log is not None:
            log.append("Ошибка: " + text)
        on_failed = ctx.get("on_failed")
        if on_failed is not None:
            try:
                on_failed(text)
            except Exception:
                pass
        self._show_error("Ошибка", text)

    def _start_worker(self, worker, on_finished, progress_bar: QProgressBar | None = None, log: QTextEdit | None = None, on_failed=None) -> None:
        thread = QThread(self)
        worker.moveToThread(thread)


        self.threads.append(thread)
        self.workers.append(worker)
        self.worker_contexts[id(worker)] = {
            "progress_bar": progress_bar,
            "log": log,
            "on_failed": on_failed,
        }

        def cleanup() -> None:
            self.worker_contexts.pop(id(worker), None)
            try:
                if worker in self.workers:
                    self.workers.remove(worker)
            except Exception:
                pass
            try:
                if thread in self.threads:
                    self.threads.remove(thread)
            except Exception:
                pass

        thread.started.connect(worker.run)
        worker.finished.connect(on_finished, Qt.QueuedConnection)
        worker.finished.connect(thread.quit, Qt.QueuedConnection)
        worker.finished.connect(worker.deleteLater, Qt.QueuedConnection)
        worker.failed.connect(self._on_worker_failed, Qt.QueuedConnection)
        worker.failed.connect(thread.quit, Qt.QueuedConnection)
        worker.failed.connect(worker.deleteLater, Qt.QueuedConnection)
        worker.progress.connect(self._on_worker_progress, Qt.QueuedConnection)
        thread.finished.connect(cleanup)
        thread.finished.connect(thread.deleteLater)

        if progress_bar:
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
        thread.start()

    def _apply_theme(self, theme: str) -> None:
        app = QApplication.instance()
        if app is None:
            return
        if theme == "dark":
            app.setStyleSheet(DARK_QSS)
        elif theme == "light":
            app.setStyleSheet(LIGHT_QSS)
        else:
            app.setStyleSheet("")


def run_app() -> None:
    import sys
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
