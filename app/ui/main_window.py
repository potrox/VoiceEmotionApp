"""Главное окно и жизненный цикл настольного приложения."""

from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QTabWidget,
    QTextEdit,
    QApplication,
)

from ..audio import stop_playback
from ..config import AppConfig, ProjectPaths
from ..dataset import DatasetLoadResult
from ..microphone import MicrophoneRecorder
from ..storage import setup_logging
from .styles import DARK_STYLESHEET, LIGHT_STYLESHEET
from .home_tab import HomeTabMixin
from .dataset_tab import DatasetTabMixin
from .recording_tab import RecordingTabMixin
from .training_tab import TrainingTabMixin
from .testing_tab import TestingTabMixin
from .recognition_tab import RecognitionTabMixin
from .server_tab import ServerTabMixin
from .settings_tab import SettingsTabMixin

class MainWindow(
    HomeTabMixin,
    DatasetTabMixin,
    RecordingTabMixin,
    TrainingTabMixin,
    TestingTabMixin,
    RecognitionTabMixin,
    ServerTabMixin,
    SettingsTabMixin,
    QMainWindow,
):

    def __init__(self):
        super().__init__()
        self.config = AppConfig.load("config.json")
        self.project_paths = ProjectPaths(self.config)
        self.project_paths.ensure()
        setup_logging(self.project_paths)
        self.dataset_result: DatasetLoadResult | None = None
        self.current_model_path: str | None = None
        self.recording_buffer = None
        self.recording_cache_path: Path | None = None
        self.is_playing_recording = False
        self.is_recording_now = False
        self.recording_generation = 0
        self.record_started_at = 0.0
        self.record_expected_duration = 0.0
        self.record_timer = QTimer(self)
        self.record_timer.setInterval(200)
        self.record_timer.timeout.connect(self._update_record_timer_progress)
        self.dataset_microphone = MicrophoneRecorder(self.config.sample_rate)
        self.is_recognition_mic_recording = False
        self.recognition_microphone = MicrophoneRecorder(self.config.sample_rate)
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
        self._build_server_tab()
        self._build_settings_tab()
        self._apply_theme(self.config.theme)
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

    def _on_worker_progress(self, value: int, text: str) -> None:
        worker = self.sender()
        worker_context = self.worker_contexts.get(id(worker), {}) if worker is not None else {}
        progress_bar = worker_context.get("progress_bar")
        operation_log = worker_context.get("log")
        if progress_bar is not None:
            progress_bar.setValue(int(value))
        if operation_log is not None:
            operation_log.append(f"[{int(value)}%] {text}")

    def _on_worker_failed(self, message: str) -> None:
        worker = self.sender()
        worker_context = self.worker_contexts.get(id(worker), {}) if worker is not None else {}
        text = str(message).strip() or "Операция завершилась ошибкой, но подробное сообщение не было передано. Проверьте logs/."
        operation_log = worker_context.get("log")
        if operation_log is not None:
            operation_log.append("Ошибка: " + text)
        on_failed = worker_context.get("on_failed")
        if on_failed is not None:
            try:
                on_failed(text)
            except Exception:
                pass
        self._show_error("Ошибка", text)

    def _start_worker(
        self,
        worker,
        on_finished,
        progress_bar: QProgressBar | None = None,
        log: QTextEdit | None = None,
        on_failed=None,
    ) -> None:
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
            app.setStyleSheet(DARK_STYLESHEET)
        elif theme == "light":
            app.setStyleSheet(LIGHT_STYLESHEET)
        else:
            app.setStyleSheet("")
