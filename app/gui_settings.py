from __future__ import annotations

import json

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)


class SettingsTabMixin:
    def _build_settings_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        form = QFormLayout()
        self.theme_combo = QComboBox(); self.theme_combo.addItems(["system", "light", "dark"]); self.theme_combo.setCurrentText(self.cfg.theme)
        self.setting_record_duration = QSpinBox(); self.setting_record_duration.setRange(3, 10); self.setting_record_duration.setValue(self.cfg.record_duration_sec)
        self.setting_test_size = QDoubleSpinBox(); self.setting_test_size.setRange(0.1, 0.5); self.setting_test_size.setSingleStep(0.05); self.setting_test_size.setValue(self.cfg.test_size)
        self.base_dir_edit = QLineEdit(self.cfg.base_dir)
        self.setting_denoise = QCheckBox("Шумоподавление"); self.setting_denoise.setChecked(self.cfg.denoise)
        self.setting_trim = QCheckBox("Удаление лишней тишины"); self.setting_trim.setChecked(self.cfg.trim_silence)
        self.setting_norm = QCheckBox("Нормализация амплитуды"); self.setting_norm.setChecked(self.cfg.normalize_amplitude)
        form.addRow("Тема интерфейса", self.theme_combo)
        form.addRow("Длительность записи", self.setting_record_duration)
        form.addRow("Процент тестовой выборки", self.setting_test_size)
        protocol_label = QLabel("Дикторы train/test не пересекаются; подбор — grouped CV, seed=42")
        protocol_label.setWordWrap(True)
        form.addRow("Протокол оценки", protocol_label)
        form.addRow("Базовая папка сохранения", self.base_dir_edit)
        form.addRow("Обработка аудио", self.setting_denoise)
        form.addRow("", self.setting_trim)
        form.addRow("", self.setting_norm)
        layout.addLayout(form)
        btn_save = QPushButton("Сохранить настройки")
        btn_save.clicked.connect(self._save_settings)
        layout.addWidget(btn_save)
        self.tabs.addTab(tab, "Настройки")

    def _save_runtime_settings(self, show_message: bool = True) -> None:
        selected = [name for name, cb in self.train_model_checks.items() if cb.isChecked()] if hasattr(self, "train_model_checks") else self.cfg.training_models
        self.cfg.training_models = selected or self.cfg.training_models
        if hasattr(self, "param_mode_combo"):
            self.cfg.parameter_mode = "auto"
        if hasattr(self, "validate_audio_checkbox"):
            self.cfg.validate_external_audio = self.validate_audio_checkbox.isChecked()
        self.cfg.save("config.json")
        if show_message:
            self._show_info("Настройки", "Настройки сохранены в config.json.")

    def _save_settings(self) -> None:
        self.cfg.theme = self.theme_combo.currentText()
        self.cfg.record_duration_sec = self.setting_record_duration.value()
        self.cfg.test_size = float(self.setting_test_size.value())
        self.cfg.base_dir = self.base_dir_edit.text().strip() or "."
        self.cfg.denoise = self.setting_denoise.isChecked()
        self.cfg.trim_silence = self.setting_trim.isChecked()
        self.cfg.normalize_amplitude = self.setting_norm.isChecked()
        self._save_runtime_settings(show_message=False)
        self._apply_theme(self.cfg.theme)
        self._show_info("Настройки", "Настройки сохранены в config.json.")
