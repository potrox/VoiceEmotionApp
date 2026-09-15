"""Вкладка загрузки и проверки датасета."""

import json
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..constants import EMOTION_RU
from ..dataset import DatasetLoadResult, inspect_emotion_labels
from ..dataset_recovery import (
    check_user_dataset_csv,
    rebuild_user_dataset_csv,
    scan_user_dataset_recordings,
)
from ..workers import DatasetLoadWorker
from .dialogs import EmotionMappingDialog, RecoveryMetadataDialog

class DatasetTabMixin:

    def _append_dataset_log(self, message: str) -> None:
        log = getattr(self, "dataset_log", None)
        if log is not None:
            log.append(message)

    def _recover_user_dataset_csv_on_startup(self) -> None:
        try:
            base = self.project_paths.datasets / "user_dataset"
            csv_path = base / "user_dataset.csv"
            records = scan_user_dataset_recordings(base, sample_rate=self.config.sample_rate)
            health = check_user_dataset_csv(csv_path, records)
            if not health.needs_rebuild:
                return
            answer = QMessageBox.question(
                self,
                "Восстановление CSV датасета",
                "В папке пользовательского датасета найдены аудиозаписи, но CSV нужно восстановить.\n\n"
                f"Причина: {health.reason}\n"
                f"Найдено аудиозаписей: {health.files_found}.\n\n"
                "Создать новый user_dataset.csv на основе имеющихся WAV/MP3-файлов?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer == QMessageBox.Yes:
                self._rebuild_user_dataset_csv_interactive(records, health.reason)
        except Exception as exc:
            self._append_dataset_log("Ошибка автоматической проверки пользовательского CSV: " + str(exc))

    def _manual_rebuild_user_dataset_csv(self) -> None:
        try:
            base = self.project_paths.datasets / "user_dataset"
            csv_path = base / "user_dataset.csv"
            records = scan_user_dataset_recordings(base, sample_rate=self.config.sample_rate)
            if not records:
                self._show_info("Записи не найдены", "В папке datasets/user_dataset пока нет WAV/MP3-файлов для восстановления CSV.")
                return
            health = check_user_dataset_csv(csv_path, records)
            reason = health.reason if health.needs_rebuild else "CSV восстановлен вручную."
            self._rebuild_user_dataset_csv_interactive(records, reason)
        except Exception as exc:
            self._show_error("Ошибка восстановления CSV", str(exc))

    def _rebuild_user_dataset_csv_interactive(self, records, reason: str) -> None:
        if not records:
            self._show_info("Записи не найдены", "В папке datasets/user_dataset пока нет WAV/MP3-файлов для восстановления CSV.")
            return
        speaker_counts: dict[str, int] = {}
        for record in records:
            speaker_counts[record.speaker_id] = speaker_counts.get(record.speaker_id, 0) + 1

        dialog = RecoveryMetadataDialog(speaker_counts, self)
        if dialog.exec() != QDialog.Accepted:
            self._append_dataset_log("Восстановление CSV отменено пользователем.")
            return

        csv_path = self.project_paths.datasets / "user_dataset" / "user_dataset.csv"
        try:
            result_path = rebuild_user_dataset_csv(csv_path, records, dialog.metadata())
        except Exception as exc:
            self._show_error("Ошибка восстановления CSV", str(exc))
            return

        self.dataset_path_edit.setText(str(result_path))
        self._refresh_record_progress()
        self._append_dataset_log(
            f"CSV пользовательского датасета восстановлен: {result_path}. "
            f"Записей: {len(records)}. Причина восстановления: {reason}"
        )
        self._show_info(
            "CSV восстановлен",
            f"Файл user_dataset.csv создан заново.\nЗаписей добавлено: {len(records)}.\nПуть: {result_path}",
        )

    def _build_dataset_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        top = QHBoxLayout()
        self.dataset_path_edit = QLineEdit()
        self.dataset_path_edit.setPlaceholderText("Путь к CSV-файлу")
        btn = QPushButton("Выбрать CSV")
        btn.clicked.connect(self._select_dataset_csv)
        btn_rebuild = QPushButton("Восстановить CSV из записей")
        btn_rebuild.clicked.connect(self._manual_rebuild_user_dataset_csv)
        top.addWidget(self.dataset_path_edit)
        top.addWidget(btn)
        top.addWidget(btn_rebuild)
        layout.addLayout(top)
        self.validate_audio_checkbox = QCheckBox("Проверять качество аудио при загрузке датасета")
        self.validate_audio_checkbox.setChecked(self.config.validate_external_audio)
        layout.addWidget(self.validate_audio_checkbox)
        self.dataset_progress = QProgressBar()
        layout.addWidget(self.dataset_progress)
        self.dataset_table = QTableWidget(0, 2)
        self.dataset_table.setHorizontalHeaderLabels(["Эмоция", "Количество"])
        self.dataset_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.dataset_table)
        self.dataset_log = QTextEdit()
        self.dataset_log.setReadOnly(True)
        layout.addWidget(self.dataset_log)
        self.tabs.addTab(tab, "Датасет")

    def _select_dataset_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выберите CSV-датасет", str(self.project_paths.datasets), "CSV (*.csv)")
        if not path:
            return
        self.dataset_path_edit.setText(path)
        try:
            infos = inspect_emotion_labels(path)
        except Exception as exc:


            try:
                selected_path = Path(path)
                parts = [part.lower() for part in selected_path.parts]
                if selected_path.name.lower() == "user_dataset.csv" and "user_dataset" in parts:
                    records = scan_user_dataset_recordings(selected_path.parent, sample_rate=self.config.sample_rate)
                    if records:
                        answer = QMessageBox.question(
                            self,
                            "CSV повреждён",
                            f"Не удалось прочитать выбранный CSV:\n{exc}\n\n"
                            f"Но в папке найдено аудиозаписей: {len(records)}. "
                            "Восстановить CSV на основе этих файлов?",
                            QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.Yes,
                        )
                        if answer == QMessageBox.Yes:
                            self._rebuild_user_dataset_csv_interactive(records, "Выбранный CSV не удалось прочитать: " + str(exc))
                        return
            except Exception:
                pass
            self._show_error("Ошибка CSV", str(exc))
            return
        dialog = EmotionMappingDialog(infos, self)
        if dialog.exec() != QDialog.Accepted:
            self.dataset_log.append("Загрузка отменена пользователем на этапе сопоставления эмоций.")
            return
        mapping = dialog.mapping()
        self.dataset_log.append("Выбранное сопоставление меток: " + json.dumps(mapping, ensure_ascii=False))
        self.dataset_log.append("Запускаю загрузку и проверку CSV-датасета...")
        self.dataset_progress.setRange(0, 100)
        self.dataset_progress.setValue(0)
        worker = DatasetLoadWorker(
            path,
            self.config.min_train_duration_sec,
            emotion_mapping=mapping,
            validate_audio_quality=self.validate_audio_checkbox.isChecked(),
            sample_rate=self.config.sample_rate,
        )
        self._start_worker(worker, self._on_dataset_loaded, self.dataset_progress, self.dataset_log)

    def _on_dataset_loaded(self, result: DatasetLoadResult) -> None:
        self.dataset_result = result
        self.dataset_table.setRowCount(0)
        for row, (emotion, count) in enumerate(result.distribution.items()):
            self.dataset_table.insertRow(row)
            self.dataset_table.setItem(row, 0, QTableWidgetItem(f"{emotion} / {EMOTION_RU.get(emotion, emotion)}"))
            self.dataset_table.setItem(row, 1, QTableWidgetItem(str(count)))
        self.dataset_log.append("Датасет успешно загружен.")
        self.dataset_log.append("Фактическое сопоставление меток: " + json.dumps(result.mapping, ensure_ascii=False))
        for warning in result.warnings:
            self.dataset_log.append("Предупреждение: " + warning)
        self._refresh_home()
        self.tabs.setCurrentIndex(1)
