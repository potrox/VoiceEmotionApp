"""Вкладка настроек аудио, обучения и интерфейса."""

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

class SettingsTabMixin:

    def _build_settings_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        form = QFormLayout()
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["system", "light", "dark"])
        self.theme_combo.setCurrentText(self.config.theme)
        self.setting_record_duration = QSpinBox()
        self.setting_record_duration.setRange(3, 10)
        self.setting_record_duration.setValue(self.config.record_duration_sec)
        self.setting_test_size = QDoubleSpinBox()
        self.setting_test_size.setRange(0.1, 0.5)
        self.setting_test_size.setSingleStep(0.05)
        self.setting_test_size.setValue(self.config.test_size)
        self.gpu_combo = QComboBox()
        self.gpu_combo.addItems(["true", "false"])
        self.gpu_combo.setCurrentText(str(self.config.use_gpu_if_available).lower())
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["quality", "balanced"])
        self.mode_combo.setCurrentText(self.config.hyperparameter_mode)
        self.base_dir_edit = QLineEdit(self.config.base_dir)
        self.setting_denoise = QCheckBox("Шумоподавление")
        self.setting_denoise.setChecked(self.config.denoise)
        self.setting_trim = QCheckBox("Удаление лишней тишины")
        self.setting_trim.setChecked(self.config.trim_silence)
        self.setting_norm = QCheckBox("Нормализация амплитуды")
        self.setting_norm.setChecked(self.config.normalize_amplitude)
        form.addRow("Тема интерфейса", self.theme_combo)
        form.addRow("Длительность записи", self.setting_record_duration)
        form.addRow("Процент тестовой выборки", self.setting_test_size)
        form.addRow("Использовать GPU при наличии", self.gpu_combo)
        form.addRow("Режим подбора параметров", self.mode_combo)
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
        selected = [name for name, cb in self.train_model_checks.items() if cb.isChecked()] if hasattr(self, "train_model_checks") else self.config.training_models
        self.config.training_models = selected or self.config.training_models
        if hasattr(self, "param_mode_combo"):
            self.config.parameter_mode = self.param_mode_combo.currentText()
            self.config.manual_svm_c = self.svm_c_spin.value()
            self.config.manual_svm_gamma = self.svm_gamma_combo.currentText()
            self.config.manual_svm_kernel = self.svm_kernel_combo.currentText()
            self.config.manual_rf_n_estimators = self.rf_n_spin.value()
            self.config.manual_rf_max_depth = self.rf_depth_spin.value()
            self.config.manual_rf_min_samples_split = self.rf_split_spin.value()
            self.config.manual_rf_min_samples_leaf = self.rf_leaf_spin.value()
            self.config.manual_mlp_hidden_layers = self.mlp_hidden_edit.text().strip()
            self.config.manual_mlp_learning_rate = self.mlp_lr_spin.value()
            self.config.manual_mlp_epochs = self.mlp_epochs_spin.value()
            self.config.manual_mlp_batch_size = self.mlp_batch_spin.value()
        if hasattr(self, "validate_audio_checkbox"):
            self.config.validate_external_audio = self.validate_audio_checkbox.isChecked()
        self.config.save("config.json")
        if show_message:
            self._show_info("Настройки", "Настройки сохранены в config.json.")

    def _save_settings(self) -> None:
        self.config.theme = self.theme_combo.currentText()
        self.config.record_duration_sec = self.setting_record_duration.value()
        self.config.test_size = float(self.setting_test_size.value())
        self.config.use_gpu_if_available = self.gpu_combo.currentText() == "true"
        self.config.hyperparameter_mode = self.mode_combo.currentText()
        self.config.base_dir = self.base_dir_edit.text().strip() or "."
        self.config.denoise = self.setting_denoise.isChecked()
        self.config.trim_silence = self.setting_trim.isChecked()
        self.config.normalize_amplitude = self.setting_norm.isChecked()
        self._save_runtime_settings(show_message=False)
        self._apply_theme(self.config.theme)
        self._show_info("Настройки", "Настройки сохранены в config.json.")
