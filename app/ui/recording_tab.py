"""Вкладка записи голосовых примеров для пользовательского датасета."""

import csv
import time
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..audio import (
    analyze_quality,
    list_input_devices,
    play_wav_file,
    save_wav,
    stop_playback,
)
from ..constants import (
    AGE_GROUPS,
    EMOTIONS,
    EMOTION_RU,
    GENDERS,
    LANGUAGES,
    STANDARD_PHRASES,
)
from ..recorder import UserDatasetRecorder
from ..microphone import EmptyRecordingError, MicrophoneError
from ..storage import timestamp

class RecordingTabMixin:

    def _build_record_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        form = QFormLayout()
        self.speaker_edit = QLineEdit()
        self.speaker_edit.setText(UserDatasetRecorder(self.project_paths.datasets, self.config.sample_rate).next_speaker_id())
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(GENDERS)
        self.age_combo = QComboBox()
        self.age_combo.addItems(AGE_GROUPS)
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(LANGUAGES)
        self.record_emotion_combo = QComboBox()
        self.record_emotion_combo.addItems(
            [f"{emotion} — {EMOTION_RU[emotion]}" for emotion in EMOTIONS]
        )
        self.phrase_combo = QComboBox()
        self.phrase_combo.addItems(STANDARD_PHRASES)
        self.phrase_combo.setEditable(True)
        self.record_duration_spin = QSpinBox()
        self.record_duration_spin.setRange(3, 10)
        self.record_duration_spin.setValue(self.config.record_duration_sec)

        self.record_device_combo = QComboBox()
        self.btn_refresh_record_devices = QPushButton("Обновить устройства")
        self.btn_refresh_record_devices.clicked.connect(self._refresh_input_devices)
        device_row = QHBoxLayout()
        device_row.addWidget(self.record_device_combo)
        device_row.addWidget(self.btn_refresh_record_devices)
        device_widget = QWidget()
        device_widget.setLayout(device_row)

        form.addRow("Диктор", self.speaker_edit)
        form.addRow("Пол", self.gender_combo)
        form.addRow("Возрастная группа", self.age_combo)
        form.addRow("Язык", self.lang_combo)
        form.addRow("Эмоция", self.record_emotion_combo)
        form.addRow("Фраза", self.phrase_combo)
        form.addRow("Длительность записи, сек.", self.record_duration_spin)
        form.addRow("Устройство записи", device_widget)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.btn_record_voice = QPushButton("Начать запись / перезаписать")
        self.btn_record_voice.clicked.connect(self._record_voice)
        self.btn_play_recording = QPushButton("Прослушать запись")
        self.btn_play_recording.clicked.connect(self._play_recording)
        self.btn_delete_recording = QPushButton("Удалить текущую запись")
        self.btn_delete_recording.clicked.connect(self._delete_recording)
        self.btn_save_recording = QPushButton("Сохранить в датасет")
        self.btn_save_recording.clicked.connect(self._save_recording)
        self.btn_import_phrases = QPushButton("Импортировать фразы из TXT")
        self.btn_import_phrases.clicked.connect(self._import_phrases)
        for b in [self.btn_record_voice, self.btn_play_recording, self.btn_delete_recording, self.btn_save_recording, self.btn_import_phrases]:
            buttons.addWidget(b)
        layout.addLayout(buttons)

        self.record_activity_bar = QProgressBar()
        self.record_activity_bar.setRange(0, 100)
        self.record_activity_bar.setValue(0)
        layout.addWidget(self.record_activity_bar)
        self.record_cache_label = QLabel("Кэш записи: пусто")
        self.record_cache_label.setWordWrap(True)
        layout.addWidget(self.record_cache_label)

        self.record_progress_table = QTableWidget(0, 2)
        self.record_progress_table.setHorizontalHeaderLabels(["Эмоция", "Записей"])
        self.record_progress_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.record_progress_table)
        self.record_log = QTextEdit()
        self.record_log.setReadOnly(True)
        layout.addWidget(self.record_log)
        self.tabs.addTab(tab, "Запись голоса")
        self._refresh_input_devices()

    def _selected_input_device_index(self) -> int | None:
        combo = getattr(self, "record_device_combo", None)
        if combo is None:
            return self.config.input_device_index
        data = combo.currentData()
        if data is None:
            return None
        try:
            return int(data)
        except Exception:
            return None

    def _fill_device_combo(self, combo: QComboBox | None) -> None:
        if combo is None:
            return
        current = combo.currentData()
        combo.clear()
        devices = list_input_devices()
        if not devices:
            combo.addItem("Микрофон не найден", None)
            combo.setEnabled(False)
            return
        combo.setEnabled(True)
        for device in devices:
            suffix = " — по умолчанию" if device.get("is_default") else ""
            sample_rate = device.get("default_samplerate") or "?"
            description = (
                f"#{device['index']} {device['name']} "
                f"({device['channels']} канал., {sample_rate} Гц){suffix}"
            )
            combo.addItem(description, int(device["index"]))
        preferred = self.config.input_device_index
        for item_index in range(combo.count()):
            if combo.itemData(item_index) == current or combo.itemData(item_index) == preferred:
                combo.setCurrentIndex(item_index)
                break

    def _refresh_input_devices(self) -> None:
        self._fill_device_combo(getattr(self, "record_device_combo", None))
        self._fill_device_combo(getattr(self, "recognition_device_combo", None))
        if hasattr(self, "record_log"):
            if self.record_device_combo.isEnabled():
                self.record_log.append("Список устройств записи обновлён.")
            else:
                self.record_log.append("Устройства записи не найдены. Проверьте подключение микрофона и разрешения Windows.")

    def _set_record_cache_status(self, text: str) -> None:
        if hasattr(self, "record_cache_label"):
            self.record_cache_label.setText(text)

    def _clear_recording_cache_file(self) -> None:
        if self.recording_cache_path:
            try:
                Path(self.recording_cache_path).unlink(missing_ok=True)
            except Exception:
                pass
        self.recording_cache_path = None

    def _set_recording_ui_state(self, active: bool) -> None:
        self.is_recording_now = active
        self.btn_record_voice.setEnabled(not active)
        self.btn_record_voice.setText("Идёт запись..." if active else "Начать запись / перезаписать")
        self.btn_delete_recording.setEnabled(True)
        self.btn_delete_recording.setText("Отменить запись" if active else "Удалить текущую запись")
        for widget_name in [
            "btn_play_recording",
            "btn_save_recording",
            "btn_import_phrases",
            "btn_refresh_record_devices",
            "record_device_combo",
            "record_duration_spin",
        ]:
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setEnabled(not active)

    def _start_record_timer_progress(self, duration_sec: int) -> None:
        self.record_expected_duration = max(float(duration_sec), 0.1)
        self.record_started_at = time.monotonic()
        self.record_activity_bar.setRange(0, 100)
        self.record_activity_bar.setValue(0)
        self.record_timer.start()

    def _stop_record_timer_progress(self, final_value: int = 100) -> None:
        if self.record_timer.isActive():
            self.record_timer.stop()
        self.record_activity_bar.setRange(0, 100)
        self.record_activity_bar.setValue(final_value)

    def _update_record_timer_progress(self) -> None:
        if not self.is_recording_now:
            return
        elapsed = max(0.0, time.monotonic() - self.record_started_at)
        percent = int(min(99, (elapsed / max(self.record_expected_duration, 0.1)) * 100))
        self.record_activity_bar.setValue(percent)
        remaining = max(0.0, self.record_expected_duration - elapsed)
        self._set_record_cache_status(
            f"Кэш записи: идёт запись... осталось примерно {remaining:.1f} сек. "
            "Файл появится в cache/ после завершения."
        )


    def _record_timeout_guard(self, generation: int) -> None:
        if generation != self.recording_generation or not self.is_recording_now:
            return
        self.record_log.append("Сработал защитный таймер записи. Принудительно завершаю поток микрофона.")
        self._finish_live_recording(generation, forced=True)

    def _record_voice(self) -> None:
        if self.is_recognition_mic_recording:
            self._show_info(
                "Запись уже идёт",
                "Сначала завершите запись во вкладке «Распознавание».",
            )
            return
        if self.is_recording_now:
            self._show_info(
                "Запись уже идёт",
                "Дождитесь завершения текущей записи или нажмите «Отменить запись».",
            )
            return

        device_index = self._selected_input_device_index()
        self.config.input_device_index = device_index
        self.config.save("config.json")
        self.recording_generation += 1
        recording_generation = self.recording_generation
        self.recording_buffer = None
        self._clear_recording_cache_file()
        duration_seconds = int(self.record_duration_spin.value())

        self.dataset_microphone.device_index = device_index
        self.dataset_microphone.target_sample_rate = self.config.sample_rate
        try:
            actual_sample_rate = self.dataset_microphone.start(duration_seconds)
        except MicrophoneError as exc:
            self._show_error("Микрофон не открыт", str(exc))
            return

        self._set_recording_ui_state(True)
        self._set_record_cache_status(
            "Кэш записи: идёт запись. После завершения временный WAV будет сохранён в cache/."
        )
        self._start_record_timer_progress(duration_seconds)
        self.record_log.append(
            f"Начата запись на {duration_seconds} сек. Говорите сейчас. "
            f"Устройство: {self.record_device_combo.currentText()}; "
            f"частота: {actual_sample_rate} Гц."
        )
        QTimer.singleShot(
            duration_seconds * 1000,
            lambda: self._finish_live_recording(recording_generation),
        )
        QTimer.singleShot(
            (duration_seconds + 3) * 1000,
            lambda: self._record_timeout_guard(recording_generation),
        )

    def _finish_live_recording(
        self, recording_generation: int, forced: bool = False
    ) -> None:
        if recording_generation != self.recording_generation or not self.is_recording_now:
            return
        self._set_recording_ui_state(False)
        self._stop_record_timer_progress(0 if forced else 100)

        try:
            recording = self.dataset_microphone.stop(
                pad_to_requested_duration=True
            )
        except EmptyRecordingError as exc:
            self.recording_buffer = None
            self._clear_recording_cache_file()
            self._set_record_cache_status(f"Кэш записи: пусто — {exc}")
            self.record_log.append(
                "Запись завершена без аудиоданных. Проверьте выбранное устройство и разрешения Windows."
            )
            return
        except MicrophoneError as exc:
            self._show_error("Ошибка записи", str(exc))
            return

        try:
            cache_path = self.project_paths.cache / f"microphone_cache_{timestamp()}.wav"
            save_wav(cache_path, recording.audio_signal, recording.sample_rate)
            quality = analyze_quality(
                recording.audio_signal, recording.sample_rate, training=False
            )
        except Exception as exc:
            self.recording_buffer = None
            self._clear_recording_cache_file()
            self._set_record_cache_status(
                "Кэш записи: пусто — не удалось сохранить временный WAV."
            )
            self._show_error("Ошибка сохранения записи", str(exc))
            return

        self.recording_buffer = recording.audio_signal
        self.recording_cache_path = Path(cache_path)
        self._set_record_cache_status(
            f"Кэш записи: есть временная запись — {self.recording_cache_path}. "
            "Нажмите «Сохранить в датасет», чтобы перенести её в datasets/, "
            "или «Удалить текущую запись»."
        )
        self.record_log.append(
            f"Запись завершена. Получено кадров: {len(recording.audio_signal)}; "
            f"длительность: {quality.duration_sec:.2f} сек.; RMS: {quality.rms:.5f}."
        )
        self.record_log.append(f"Временный файл в кэше: {self.recording_cache_path}")
        if recording.driver_warnings:
            self.record_log.append(
                "Предупреждения аудиодрайвера: " + "; ".join(recording.driver_warnings)
            )
        for warning in quality.warnings:
            self.record_log.append("Предупреждение: " + warning)
        if not quality.warnings:
            self.record_log.append("Запись готова к прослушиванию и сохранению.")

    def _cancel_live_recording(self) -> None:
        if not self.is_recording_now:
            return
        self.recording_generation += 1
        self.dataset_microphone.cancel()
        self._set_recording_ui_state(False)
        self._stop_record_timer_progress(0)
        self.recording_buffer = None
        self._clear_recording_cache_file()
        self._set_record_cache_status("Кэш записи: пусто — запись отменена пользователем.")
        self.record_log.append("Активная запись отменена. Временный файл удалён.")

    def _play_recording(self) -> None:
        if self.is_recording_now:
            self._show_info("Запись идёт", "Прослушивание будет доступно после завершения записи.")
            return
        if self.recording_buffer is None and not self.recording_cache_path:
            self._show_error("Нет записи", "Сначала выполните запись с микрофона.")
            return



        if self.is_playing_recording:
            stop_playback()
            self.is_playing_recording = False
            self.btn_play_recording.setText("Прослушать запись")
            self.record_log.append("Прослушивание остановлено.")
            return

        play_path = None
        if self.recording_cache_path and Path(self.recording_cache_path).exists():
            play_path = Path(self.recording_cache_path)
        elif self.recording_buffer is not None:
            try:
                play_path = self.project_paths.cache / "microphone_cache_preview.wav"
                save_wav(play_path, self.recording_buffer, self.config.sample_rate)
                self.recording_cache_path = play_path
                self._set_record_cache_status(f"Кэш записи: есть временная запись — {play_path}")
            except Exception as exc:
                self._show_error("Ошибка прослушивания", f"Не удалось подготовить WAV-файл для прослушивания: {exc}")
                return

        if not play_path:
            self._show_error("Ошибка прослушивания", "Не найден временный файл записи для прослушивания.")
            return

        try:
            play_wav_file(play_path, wait=False)
        except Exception as exc:
            self._show_error("Ошибка прослушивания", str(exc))
            self.record_log.append("Ошибка прослушивания: " + str(exc))
            return

        self.is_playing_recording = True
        self.btn_play_recording.setText("Остановить прослушивание")
        self.record_log.append(f"Прослушивание запущено из WAV-файла в кэше: {play_path}")
        duration_ms = max(1000, int(float(self.record_duration_spin.value()) * 1000) + 300)
        QTimer.singleShot(duration_ms, self._finish_recording_playback_state)

    def _finish_recording_playback_state(self) -> None:
        if not self.is_playing_recording:
            return
        self.is_playing_recording = False
        self.btn_play_recording.setText("Прослушать запись")
        self.record_log.append("Прослушивание завершено или остановлено системой воспроизведения.")

    def _delete_recording(self) -> None:
        if self.is_recording_now:
            self._cancel_live_recording()
            return
        stop_playback()
        self.is_playing_recording = False
        self.btn_play_recording.setText("Прослушать запись")
        self.recording_buffer = None
        self._clear_recording_cache_file()
        if hasattr(self, "record_activity_bar"):
            self.record_activity_bar.setRange(0, 100)
            self.record_activity_bar.setValue(0)
        self._set_record_cache_status("Кэш записи: пусто")
        self.record_log.append("Текущая несохранённая запись удалена из буфера и кэша.")

    def _save_recording(self) -> None:
        if self.is_recording_now:
            self._show_info("Запись идёт", "Сохранение будет доступно после завершения записи.")
            return
        if self.recording_buffer is None:
            self._show_error("Нет записи", "Сначала выполните запись с микрофона.")
            return
        emotion = EMOTIONS[self.record_emotion_combo.currentIndex()]
        recorder = UserDatasetRecorder(self.project_paths.datasets, self.config.sample_rate)
        try:
            saved = recorder.save_recording(
                self.recording_buffer,
                self.speaker_edit.text().strip(),
                emotion,
                self.gender_combo.currentText(),
                self.age_combo.currentText(),
                self.phrase_combo.currentText(),
                self.lang_combo.currentText(),
            )
            self.record_log.append(f"Сохранено original: {saved.original_path}")
            self.record_log.append(f"Сохранено processed: {saved.processed_path}")
            self.record_log.append(f"CSV пользовательского датасета: {saved.csv_path}")
            stop_playback()
            self.is_playing_recording = False
            self.btn_play_recording.setText("Прослушать запись")
            self.recording_buffer = None
            self._clear_recording_cache_file()
            self.record_activity_bar.setRange(0, 100)
            self.record_activity_bar.setValue(0)
            self._set_record_cache_status("Кэш записи: пусто — запись сохранена в пользовательский датасет.")
            self._refresh_record_progress()
        except Exception as exc:
            self._show_error("Запись отклонена", str(exc))

    def _import_phrases(self) -> None:
        if self.is_recording_now:
            self._show_info("Запись идёт", "Импорт фраз будет доступен после завершения записи.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Выберите TXT с фразами", str(self.project_paths.datasets), "Text (*.txt)")
        if not path:
            return
        try:
            lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        except UnicodeDecodeError:
            lines = [line.strip() for line in Path(path).read_text(encoding="cp1251").splitlines() if line.strip()]
        existing = {
            self.phrase_combo.itemText(item_index)
            for item_index in range(self.phrase_combo.count())
        }
        added = 0
        for line in lines:
            if line not in existing:
                self.phrase_combo.addItem(line)
                existing.add(line)
                added += 1
        self.record_log.append(f"Импортировано новых фраз: {added}")

    def _refresh_record_progress(self) -> None:
        csv_path = self.project_paths.datasets / "user_dataset" / "user_dataset.csv"
        counts = {emotion: 0 for emotion in EMOTIONS}
        total = 0
        if csv_path.exists():
            try:
                with csv_path.open("r", encoding="utf-8", newline="") as f:
                    for row in csv.DictReader(f):
                        emotion = row.get("emotion", "")
                        if emotion in counts:
                            counts[emotion] += 1
                            total += 1
            except Exception:
                pass
        if hasattr(self, "record_progress_table"):
            self.record_progress_table.setRowCount(0)
            for row, emotion in enumerate(EMOTIONS):
                self.record_progress_table.insertRow(row)
                self.record_progress_table.setItem(row, 0, QTableWidgetItem(f"{emotion} / {EMOTION_RU[emotion]}"))
                self.record_progress_table.setItem(row, 1, QTableWidgetItem(str(counts[emotion])))
