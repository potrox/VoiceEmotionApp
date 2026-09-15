"""Вкладка распознавания эмоций в файлах и записи с микрофона."""

import time

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..audio import save_wav
from ..constants import EMOTION_RU
from ..features import FeatureExtractor
from ..microphone import EmptyRecordingError, MicrophoneError
from ..recognizer import Recognizer
from ..storage import timestamp
from ..workers import FunctionWorker
from .constants import MODEL_OPTIONS

class RecognitionTabMixin:

    def _build_recognition_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        model_row = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Путь к сохранённой модели .pkl")
        btn_model = QPushButton("Загрузить модель")
        btn_model.clicked.connect(self._select_model)
        self.recognition_model_combo = QComboBox()
        self.recognition_model_combo.addItems(["Лучшая модель"] + MODEL_OPTIONS)
        model_row.addWidget(self.model_path_edit)
        model_row.addWidget(btn_model)
        model_row.addWidget(QLabel("Использовать:"))
        model_row.addWidget(self.recognition_model_combo)
        layout.addLayout(model_row)

        mic_row = QHBoxLayout()
        self.recognition_device_combo = QComboBox()
        btn_refresh_recognition_devices = QPushButton("Обновить устройства записи")
        btn_refresh_recognition_devices.clicked.connect(self._refresh_input_devices)
        mic_row.addWidget(QLabel("Микрофон:"))
        mic_row.addWidget(self.recognition_device_combo)
        mic_row.addWidget(btn_refresh_recognition_devices)
        layout.addLayout(mic_row)
        self._refresh_input_devices()

        buttons = QHBoxLayout()
        self.btn_recognize_file = QPushButton("Распознать WAV/MP3-файл")
        self.btn_recognize_file.clicked.connect(self._recognize_file_dialog)
        self.btn_recognize_folder = QPushButton("Пакетное распознавание папки")
        self.btn_recognize_folder.clicked.connect(self._recognize_folder_dialog)
        self.btn_recognize_mic = QPushButton("Начать запись с микрофона")
        self.btn_recognize_mic.clicked.connect(self._recognize_mic)
        buttons.addWidget(self.btn_recognize_file)
        buttons.addWidget(self.btn_recognize_folder)
        buttons.addWidget(self.btn_recognize_mic)
        layout.addLayout(buttons)
        self.recognition_mic_status_label = QLabel("Запись с микрофона: не запущена. Нажмите кнопку, говорите сколько нужно, затем нажмите «Закончить запись».")
        self.recognition_mic_status_label.setWordWrap(True)
        layout.addWidget(self.recognition_mic_status_label)
        self.recognition_result_label = QLabel("Итоговая эмоция: —")
        self.recognition_result_label.setStyleSheet("font-size: 24px; font-weight: 600;")
        layout.addWidget(self.recognition_result_label)
        self.prob_table = QTableWidget(0, 2)
        self.prob_table.setHorizontalHeaderLabels(["Эмоция", "Вероятность"])
        self.prob_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.prob_table)
        self.segment_table = QTableWidget(0, 4)
        self.segment_table.setHorizontalHeaderLabels(["Отрезок", "Время", "Эмоция", "Уверенность"])
        self.segment_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.segment_table)
        self.recognition_log = QTextEdit()
        self.recognition_log.setReadOnly(True)
        layout.addWidget(self.recognition_log)
        self.tabs.addTab(tab, "Распознавание")

    def _select_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выберите модель", str(self.project_paths.models), "Model (*.pkl)")
        if path:
            self.current_model_path = path
            self.model_path_edit.setText(path)
            self._update_recognition_model_choices()
            self._prepare_publish_fields(path)
            self._refresh_home()

    def _update_recognition_model_choices(self) -> None:
        if not hasattr(self, "recognition_model_combo"):
            return
        try:
            recognizer = self._make_recognizer()
            available = recognizer.available_models()
        except Exception:
            available = MODEL_OPTIONS
        current = self.recognition_model_combo.currentText()
        self.recognition_model_combo.clear()
        self.recognition_model_combo.addItem("Лучшая модель")
        self.recognition_model_combo.addItems(available)
        available_models = [
            self.recognition_model_combo.itemText(item_index)
            for item_index in range(self.recognition_model_combo.count())
        ]
        if current in available_models:
            self.recognition_model_combo.setCurrentText(current)

    def _make_recognizer(self) -> Recognizer:
        model_path = self.current_model_path or self.model_path_edit.text().strip()
        if not model_path:
            raise RuntimeError("Модель не загружена. Сначала обучите или выберите сохранённую модель.")
        extractor = FeatureExtractor(self.project_paths.cache, self.config.sample_rate, self.config.denoise, self.config.trim_silence, self.config.normalize_amplitude)
        return Recognizer(model_path, extractor)

    def _selected_recognition_model(self) -> str | None:
        name = self.recognition_model_combo.currentText()
        return None if name == "Лучшая модель" else name

    def _recognize_file_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выберите аудиофайл", str(self.project_paths.records), "Audio (*.wav *.mp3)")
        if not path:
            return
        try:
            recognizer = self._make_recognizer()
        except Exception as exc:
            self._show_error("Ошибка", str(exc))
            return
        worker = FunctionWorker(recognizer.recognize_file, path, self._selected_recognition_model())
        self._start_worker(worker, self._on_recognition_result, None, self.recognition_log)
        self.tabs.setCurrentIndex(5)

    def _recognize_folder_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с аудиофайлами", str(self.project_paths.records))
        if not folder:
            return
        try:
            recognizer = self._make_recognizer()
        except Exception as exc:
            self._show_error("Ошибка", str(exc))
            return
        worker = FunctionWorker(recognizer.batch_recognize, folder, self.project_paths.exports, self._selected_recognition_model())
        def done(path):
            self.recognition_log.append(f"Пакетное распознавание завершено. CSV: {path}")
        self._start_worker(worker, done, None, self.recognition_log)
        self.tabs.setCurrentIndex(5)

    def _set_recognition_mic_ui_state(self, active: bool) -> None:
        self.is_recognition_mic_recording = active
        if hasattr(self, "btn_recognize_mic"):
            self.btn_recognize_mic.setText("Закончить запись" if active else "Начать запись с микрофона")
            self.btn_recognize_mic.setEnabled(True)
        for widget_name in [
            "btn_recognize_file",
            "btn_recognize_folder",
            "recognition_device_combo",
            "recognition_model_combo",
            "model_path_edit",
        ]:
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setEnabled(not active)


    def _update_recognition_mic_timer(self) -> None:
        if not self.is_recognition_mic_recording:
            return
        elapsed = max(0.0, time.monotonic() - self.recognition_mic_started_at)
        text = (
            f"Запись с микрофона идёт: {elapsed:.1f} сек. "
            "Нажмите «Закончить запись», чтобы остановить запись и запустить анализ по отрезкам по 5 секунд."
        )
        if hasattr(self, "recognition_mic_status_label"):
            self.recognition_mic_status_label.setText(text)

    def _cancel_recognition_mic_recording(self) -> None:
        if not self.is_recognition_mic_recording:
            return
        self.recognition_microphone.cancel()
        if self.recognition_mic_timer.isActive():
            self.recognition_mic_timer.stop()
        self._set_recognition_mic_ui_state(False)
        self.recognition_mic_status_label.setText("Запись с микрофона: отменена.")

    def _recognize_mic(self) -> None:
        if self.is_recognition_mic_recording:
            self._finish_recognition_mic_recording()
            return
        if self.is_recording_now:
            self._show_info(
                "Запись уже идёт",
                "Сначала завершите запись во вкладке «Запись голоса».",
            )
            return
        try:
            self._make_recognizer()
        except Exception as exc:
            self._show_error("Ошибка", str(exc))
            return

        device_index = self.recognition_device_combo.currentData()
        device_index = int(device_index) if device_index is not None else None
        self.config.input_device_index = device_index
        self.config.save("config.json")
        self.recognition_microphone.device_index = device_index
        self.recognition_microphone.target_sample_rate = self.config.sample_rate
        try:
            actual_sample_rate = self.recognition_microphone.start()
        except MicrophoneError as exc:
            self._show_error("Микрофон не открыт", str(exc))
            return

        self._set_recognition_mic_ui_state(True)
        self.recognition_mic_started_at = time.monotonic()
        self.recognition_mic_timer.start()
        self.segment_table.setRowCount(0)
        self.prob_table.setRowCount(0)
        self.recognition_result_label.setText("Итоговая эмоция: запись идёт...")
        self.recognition_log.append(
            "Начата свободная запись с микрофона. "
            "Для завершения нажмите «Закончить запись». "
            f"Устройство: {self.recognition_device_combo.currentText()}; "
            f"частота: {actual_sample_rate} Гц."
        )
        self.tabs.setCurrentIndex(5)

    def _finish_recognition_mic_recording(self) -> None:
        if not self.is_recognition_mic_recording:
            return
        elapsed_seconds = max(
            0.0, time.monotonic() - self.recognition_mic_started_at
        )
        if self.recognition_mic_timer.isActive():
            self.recognition_mic_timer.stop()
        self._set_recognition_mic_ui_state(False)

        try:
            recording = self.recognition_microphone.stop()
        except EmptyRecordingError as exc:
            self.recognition_result_label.setText("Итоговая эмоция: —")
            self.recognition_mic_status_label.setText(
                f"Запись с микрофона: пусто — {exc}"
            )
            self.recognition_log.append(
                "Запись завершена без аудиоданных. Проверьте выбранное устройство."
            )
            return
        except MicrophoneError as exc:
            self.recognition_result_label.setText("Итоговая эмоция: —")
            self._show_error("Ошибка записи", str(exc))
            return

        try:
            temporary_path = (
                self.project_paths.cache / f"mic_recognition_long_{timestamp()}.wav"
            )
            save_wav(
                temporary_path, recording.audio_signal, recording.sample_rate
            )
        except Exception as exc:
            self.recognition_result_label.setText("Итоговая эмоция: —")
            self._show_error(
                "Ошибка записи",
                f"Не удалось сохранить запись с микрофона: {exc}",
            )
            return

        duration_seconds = recording.duration_seconds or elapsed_seconds
        self.recognition_mic_status_label.setText(
            f"Запись завершена: {duration_seconds:.1f} сек. "
            "Выполняется анализ по отрезкам по 5 секунд..."
        )
        self.recognition_log.append(
            f"Запись завершена. Длительность: {duration_seconds:.2f} сек. "
            f"Временный файл: {temporary_path}"
        )
        if recording.driver_warnings:
            self.recognition_log.append(
                "Предупреждения аудиодрайвера: "
                + "; ".join(recording.driver_warnings)
            )
        self.recognition_result_label.setText(
            "Итоговая эмоция: выполняется анализ..."
        )

        try:
            recognizer = self._make_recognizer()
            worker = FunctionWorker(
                recognizer.recognize_file_segmented,
                temporary_path,
                self._selected_recognition_model(),
                5.0,
            )
            self._start_worker(
                worker,
                self._on_recognition_result,
                None,
                self.recognition_log,
            )
        except Exception as exc:
            self.recognition_result_label.setText("Итоговая эмоция: —")
            self._show_error("Ошибка распознавания", str(exc))

    def _on_recognition_result(self, result) -> None:
        if result.error:
            self.recognition_result_label.setText("Итоговая эмоция: —")
            self.recognition_log.append("Ошибка распознавания: " + result.error)
            return
        is_segmented = hasattr(result, "segments")
        if is_segmented:
            self.recognition_result_label.setText(
                f"Общий эмоциональный фон: {result.predicted_emotion_ru} ({result.confidence:.2%})"
            )
        else:
            self.recognition_result_label.setText(f"Итоговая эмоция: {result.predicted_emotion_ru} ({result.confidence:.2%})")
        self.prob_table.setRowCount(0)
        for row, (emotion, prob) in enumerate(sorted(result.probabilities.items())):
            self.prob_table.insertRow(row)
            self.prob_table.setItem(row, 0, QTableWidgetItem(f"{emotion} / {EMOTION_RU.get(emotion, emotion)}"))
            self.prob_table.setItem(row, 1, QTableWidgetItem(f"{prob:.4f}"))
        if hasattr(self, "segment_table"):
            self.segment_table.setRowCount(0)
            if is_segmented:
                for row, seg in enumerate(result.segments):
                    self.segment_table.insertRow(row)
                    self.segment_table.setItem(row, 0, QTableWidgetItem(str(seg.index)))
                    self.segment_table.setItem(row, 1, QTableWidgetItem(f"{seg.start_sec:.1f}–{seg.end_sec:.1f} сек."))
                    self.segment_table.setItem(row, 2, QTableWidgetItem(f"{seg.predicted_emotion} / {seg.predicted_emotion_ru}"))
                    self.segment_table.setItem(row, 3, QTableWidgetItem(f"{seg.confidence:.2%}"))
        self.recognition_log.append(f"Файл: {result.file_path}")
        self.recognition_log.append(f"Модель: {result.model_name}; время обработки: {result.processing_time_sec:.3f} сек.")
        if is_segmented:
            self.recognition_log.append(
                f"Сегментный анализ: {len(result.segments)} отрезк(ов) по {result.segment_duration_sec:.0f} сек.; "
                "общий эмоциональный фон рассчитан как среднее распределение вероятностей по всем отрезкам."
            )
            for seg in result.segments:
                self.recognition_log.append(
                    f"Отрезок {seg.index}: {seg.start_sec:.1f}–{seg.end_sec:.1f} сек. → "
                    f"{seg.predicted_emotion_ru} ({seg.confidence:.2%})"
                )
                for w in seg.warnings:
                    self.recognition_log.append(f"Предупреждение по отрезку {seg.index}: {w}")
            if hasattr(self, "recognition_mic_status_label"):
                self.recognition_mic_status_label.setText(
                    f"Анализ завершён: общий эмоциональный фон — {result.predicted_emotion_ru} ({result.confidence:.2%})."
                )
        for w in result.warnings:
            self.recognition_log.append("Предупреждение: " + w)
