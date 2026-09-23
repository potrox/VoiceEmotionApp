from __future__ import annotations

from pathlib import Path
import json
import time

from PySide6.QtWidgets import (
    QCheckBox,
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
import numpy as np

from .audio import save_wav, _resample_if_needed
from .constants import EMOTION_RU
from .features import FeatureExtractor
from .gui_shared import MODEL_OPTIONS
from .recognizer import Recognizer
from .storage import timestamp
from .workers import FunctionWorker


class RecognitionTabMixin:
    def _build_recognition_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        model_row = QHBoxLayout()
        self.model_path_edit = QLineEdit(); self.model_path_edit.setPlaceholderText("Путь к сохранённой модели .pkl")
        btn_model = QPushButton("Загрузить модель")
        btn_model.clicked.connect(self._select_model)
        self.recognition_model_combo = QComboBox()
        self.recognition_model_combo.addItems(["Лучшая модель"] + MODEL_OPTIONS)
        model_row.addWidget(self.model_path_edit)
        model_row.addWidget(btn_model)
        model_row.addWidget(QLabel("Использовать:"))
        model_row.addWidget(self.recognition_model_combo)
        layout.addLayout(model_row)

        transcript_row = QHBoxLayout()
        self.recognition_transcript_edit = QLineEdit()
        self.recognition_transcript_edit.setPlaceholderText(
            "Необязательно: ручной текст заменяет авторасшифровку файла"
        )
        transcript_row.addWidget(QLabel("Транскрипт:"))
        transcript_row.addWidget(self.recognition_transcript_edit)
        layout.addLayout(transcript_row)
        self.recognition_auto_asr_check = QCheckBox("Создавать транскрипт автоматически (локально, CPU)")
        self.recognition_auto_asr_check.setChecked(True)
        self.recognition_auto_asr_check.setToolTip(
            "При первом запуске скачиваются веса модели ASR (около 461 МиБ). "
            "Аудио обрабатывается локально; ручной текст для файла имеет приоритет."
        )
        layout.addWidget(self.recognition_auto_asr_check)
        self.recognition_transcript_result = QLabel("Распознанный текст: —")
        self.recognition_transcript_result.setWordWrap(True)
        layout.addWidget(self.recognition_transcript_result)

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
        buttons.addWidget(self.btn_recognize_file); buttons.addWidget(self.btn_recognize_folder); buttons.addWidget(self.btn_recognize_mic)
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
        self.recognition_log = QTextEdit(); self.recognition_log.setReadOnly(True)
        layout.addWidget(self.recognition_log)
        self.tabs.addTab(tab, "Распознавание")

    def _load_best_model_pointer(self) -> None:
        # Do not auto-load a legacy model trained with the old leaky protocol.
        pointer = self.paths.models / "best_model_honest.txt"
        if pointer.exists():
            model_path = pointer.read_text(encoding="utf-8").strip()
            if model_path and Path(model_path).exists():
                self.current_model_path = model_path
                if hasattr(self, "model_path_edit"):
                    self.model_path_edit.setText(model_path)
                    self._update_recognition_model_choices()
                return
        bundled_model = Path(__file__).resolve().parents[1] / "models" / "model_v11_dusha_62k.pkl"
        if bundled_model.is_file():
            self.current_model_path = str(bundled_model)
            if hasattr(self, "model_path_edit"):
                self.model_path_edit.setText(self.current_model_path)
                self._update_recognition_model_choices()

    def _select_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выберите модель", str(self.paths.models), "Model (*.pkl)")
        if path:
            self.current_model_path = path
            self.model_path_edit.setText(path)
            self._update_recognition_model_choices()
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
        if current in [self.recognition_model_combo.itemText(i) for i in range(self.recognition_model_combo.count())]:
            self.recognition_model_combo.setCurrentText(current)

    def _make_recognizer(self) -> Recognizer:
        model_path = self.current_model_path or self.model_path_edit.text().strip()
        if not model_path:
            raise RuntimeError("Модель не загружена. Сначала обучите или выберите сохранённую модель.")
        resolved_model_path = str(Path(model_path).resolve())
        if (
            self._recognizer_cache is not None
            and self._recognizer_cache_path == resolved_model_path
        ):
            return self._recognizer_cache
        extractor = FeatureExtractor(self.paths.cache, self.cfg.sample_rate, self.cfg.denoise, self.cfg.trim_silence, self.cfg.normalize_amplitude)
        recognizer = Recognizer(model_path, extractor)
        self._recognizer_cache = recognizer
        self._recognizer_cache_path = resolved_model_path
        return recognizer

    def _selected_recognition_model(self) -> str | None:
        name = self.recognition_model_combo.currentText()
        return None if name == "Лучшая модель" else name

    def _recognize_file_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выберите аудиофайл", str(self.paths.records), "Audio (*.wav *.mp3)")
        if not path:
            return
        try:
            recognizer = self._make_recognizer()
        except Exception as exc:
            self._show_error("Ошибка", str(exc)); return
        transcript = self.recognition_transcript_edit.text().strip()
        self.recognition_result_label.setText("Итоговая эмоция: расшифровка и анализ...")
        worker = FunctionWorker(
            recognizer.recognize_file,
            path,
            self._selected_recognition_model(),
            transcript,
            self.recognition_auto_asr_check.isChecked(),
        )
        self._start_worker(worker, self._on_recognition_result, None, self.recognition_log)
        self.tabs.setCurrentIndex(5)

    def _recognize_folder_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с аудиофайлами", str(self.paths.records))
        if not folder:
            return
        try:
            recognizer = self._make_recognizer()
        except Exception as exc:
            self._show_error("Ошибка", str(exc)); return
        worker = FunctionWorker(recognizer.batch_recognize, folder, self.paths.exports, self._selected_recognition_model(), self.recognition_auto_asr_check.isChecked())
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
            "recognition_transcript_edit",
            "recognition_auto_asr_check",
        ]:
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setEnabled(not active)

    def _close_recognition_mic_stream(self) -> None:
        stream = getattr(self, "recognition_mic_stream", None)
        self.recognition_mic_stream = None
        if stream is None:
            return
        try:
            stream.abort()
        except Exception:
            try:
                stream.stop()
            except Exception:
                pass
        try:
            stream.close()
        except Exception:
            pass

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
        self._close_recognition_mic_stream()
        self.recognition_mic_blocks = []
        self.recognition_mic_statuses = []
        if self.recognition_mic_timer.isActive():
            self.recognition_mic_timer.stop()
        self._set_recognition_mic_ui_state(False)
        if hasattr(self, "recognition_mic_status_label"):
            self.recognition_mic_status_label.setText("Запись с микрофона: отменена.")

    def _recognize_mic(self) -> None:
        if self.is_recognition_mic_recording:
            self._finish_recognition_mic_recording()
            return
        if self.is_recording_now:
            self._show_info("Запись уже идёт", "Сначала завершите запись во вкладке «Запись голоса».")
            return
        try:
            self._make_recognizer()
        except Exception as exc:
            self._show_error("Ошибка", str(exc))
            return

        device_index = None
        if hasattr(self, "recognition_device_combo"):
            data = self.recognition_device_combo.currentData()
            device_index = int(data) if data is not None else None
        self.cfg.input_device_index = device_index
        self.cfg.save("config.json")

        try:
            import sounddevice as sd
        except Exception as exc:
            self._show_error("Микрофон недоступен", f"Библиотека sounddevice не установлена или не может открыть аудиоустройства. Подробности: {exc}")
            return

        device = None if device_index is None or int(device_index) < 0 else int(device_index)
        actual_sr = int(self.cfg.sample_rate)
        try:
            sd.check_input_settings(device=device, samplerate=actual_sr, channels=1)
        except Exception:
            try:
                dev_info = sd.query_devices(device, "input") if device is not None else sd.query_devices(kind="input")
                actual_sr = int(float(dev_info.get("default_samplerate", self.cfg.sample_rate)) or self.cfg.sample_rate)
                sd.check_input_settings(device=device, samplerate=actual_sr, channels=1)
            except Exception as exc:
                self._show_error(
                    "Микрофон не открыт",
                    "Не удалось открыть выбранное устройство записи. Проверьте разрешение Windows на доступ к микрофону "
                    f"или выберите другое устройство. Подробности: {exc}",
                )
                return

        self.recognition_mic_blocks = []
        self.recognition_mic_statuses = []
        self.recognition_mic_actual_sr = actual_sr

        def audio_callback(indata, frames, _time_info, status):
            if status:
                text = str(status)
                if text and text not in self.recognition_mic_statuses:
                    self.recognition_mic_statuses.append(text)
            if not self.is_recognition_mic_recording:
                return
            if frames > 0:
                self.recognition_mic_blocks.append(np.array(indata[:frames, 0], dtype=np.float32, copy=True))

        try:
            self.recognition_mic_stream = sd.InputStream(
                samplerate=actual_sr,
                channels=1,
                dtype="float32",
                device=device,
                callback=audio_callback,
            )
            self.recognition_mic_stream.start()
        except Exception as exc:
            self.recognition_mic_stream = None
            self._show_error(
                "Ошибка записи",
                "Не удалось запустить поток микрофона. Проверьте, не занят ли микрофон другой программой. "
                f"Подробности: {exc}",
            )
            return

        self._set_recognition_mic_ui_state(True)
        self.recognition_mic_started_at = time.monotonic()
        self.recognition_mic_timer.start()
        if hasattr(self, "segment_table"):
            self.segment_table.setRowCount(0)
        self.prob_table.setRowCount(0)
        self.recognition_result_label.setText("Итоговая эмоция: запись идёт...")
        device_text = self.recognition_device_combo.currentText() if hasattr(self, "recognition_device_combo") else "по умолчанию"
        self.recognition_log.append(
            "Начата свободная запись с микрофона. "
            "Для завершения нажмите «Закончить запись». "
            f"Устройство: {device_text}; частота: {actual_sr} Гц."
        )
        self.tabs.setCurrentIndex(5)

    def _finish_recognition_mic_recording(self) -> None:
        if not self.is_recognition_mic_recording:
            return
        elapsed = max(0.0, time.monotonic() - self.recognition_mic_started_at)
        self._close_recognition_mic_stream()
        if self.recognition_mic_timer.isActive():
            self.recognition_mic_timer.stop()
        self._set_recognition_mic_ui_state(False)

        blocks = list(self.recognition_mic_blocks)
        self.recognition_mic_blocks = []
        if not blocks:
            self.recognition_result_label.setText("Итоговая эмоция: —")
            if hasattr(self, "recognition_mic_status_label"):
                self.recognition_mic_status_label.setText("Запись с микрофона: пусто — микрофон не передал аудиоданные.")
            self.recognition_log.append("Запись с микрофона завершена без аудиоданных. Проверьте выбранное устройство.")
            return

        try:
            y = np.concatenate(blocks).astype(np.float32)
            y = _resample_if_needed(y, self.recognition_mic_actual_sr, self.cfg.sample_rate)
            temp_path = self.paths.cache / f"mic_recognition_long_{timestamp()}.wav"
            save_wav(temp_path, y, self.cfg.sample_rate)
        except Exception as exc:
            self.recognition_result_label.setText("Итоговая эмоция: —")
            self._show_error("Ошибка записи", f"Не удалось сохранить запись с микрофона: {exc}")
            return

        duration = float(len(y) / self.cfg.sample_rate) if self.cfg.sample_rate else elapsed
        if hasattr(self, "recognition_mic_status_label"):
            self.recognition_mic_status_label.setText(
                f"Запись завершена: {duration:.1f} сек. Файл сохранён в cache/. Выполняется анализ по отрезкам по 5 секунд..."
            )
        self.recognition_log.append(f"Запись завершена. Длительность: {duration:.2f} сек. Временный файл: {temp_path}")
        if self.recognition_mic_statuses:
            self.recognition_log.append("Предупреждения аудиодрайвера: " + "; ".join(self.recognition_mic_statuses))
        self.recognition_result_label.setText("Итоговая эмоция: выполняется анализ...")

        try:
            recognizer = self._make_recognizer()
        except Exception as exc:
            self._show_error("Ошибка", str(exc))
            return
        worker = FunctionWorker(recognizer.recognize_file_segmented, temp_path, self._selected_recognition_model(), 5.0, self.recognition_auto_asr_check.isChecked())
        self._start_worker(worker, self._on_recognition_result, None, self.recognition_log)
        self.tabs.setCurrentIndex(5)

    def _on_recognition_result(self, result) -> None:
        if result.error:
            self.recognition_result_label.setText("Итоговая эмоция: —")
            self.recognition_transcript_result.setText("Распознанный текст: —")
            self.recognition_log.append("Ошибка распознавания: " + result.error)
            return
        source = f" ({result.transcript_source})" if result.transcript_source else ""
        self.recognition_transcript_result.setText(
            f"Распознанный текст{source}: {result.transcript or '—'}"
        )
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
