"""Вкладка настройки и запуска обучения моделей."""

import json
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
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

from ..server_integration import publish_model
from ..storage import ExperimentDB, timestamp
from ..workers import FunctionWorker, TrainingWorker
from .constants import MODEL_OPTIONS

class TrainingTabMixin:

    def _build_training_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        model_group = QGroupBox("Выбор моделей для обучения")
        model_layout = QHBoxLayout(model_group)
        self.train_model_checks: dict[str, QCheckBox] = {}
        for name in MODEL_OPTIONS:
            model_checkbox = QCheckBox(name)
            model_checkbox.setChecked(name in self.config.training_models)
            self.train_model_checks[name] = model_checkbox
            model_layout.addWidget(model_checkbox)
        layout.addWidget(model_group)

        params_group = QGroupBox("Параметры обучения")
        params_layout = QFormLayout(params_group)
        self.param_mode_combo = QComboBox()
        self.param_mode_combo.addItems(["auto", "manual"])
        self.param_mode_combo.setCurrentText(self.config.parameter_mode)
        self.svm_c_spin = QDoubleSpinBox()
        self.svm_c_spin.setRange(0.001, 1000)
        self.svm_c_spin.setValue(self.config.manual_svm_c)
        self.svm_c_spin.setDecimals(3)
        self.svm_gamma_combo = QComboBox()
        self.svm_gamma_combo.addItems(["scale", "auto"])
        self.svm_gamma_combo.setCurrentText(self.config.manual_svm_gamma)
        self.svm_kernel_combo = QComboBox()
        self.svm_kernel_combo.addItems(["rbf", "linear", "poly", "sigmoid"])
        self.svm_kernel_combo.setCurrentText(self.config.manual_svm_kernel)
        self.rf_n_spin = QSpinBox()
        self.rf_n_spin.setRange(10, 2000)
        self.rf_n_spin.setValue(self.config.manual_rf_n_estimators)
        self.rf_depth_spin = QSpinBox()
        self.rf_depth_spin.setRange(0, 200)
        self.rf_depth_spin.setValue(self.config.manual_rf_max_depth)
        self.rf_split_spin = QSpinBox()
        self.rf_split_spin.setRange(2, 50)
        self.rf_split_spin.setValue(self.config.manual_rf_min_samples_split)
        self.rf_leaf_spin = QSpinBox()
        self.rf_leaf_spin.setRange(1, 50)
        self.rf_leaf_spin.setValue(self.config.manual_rf_min_samples_leaf)
        self.mlp_hidden_edit = QLineEdit(self.config.manual_mlp_hidden_layers)
        self.mlp_lr_spin = QDoubleSpinBox()
        self.mlp_lr_spin.setRange(0.00001, 1.0)
        self.mlp_lr_spin.setDecimals(5)
        self.mlp_lr_spin.setSingleStep(0.0005)
        self.mlp_lr_spin.setValue(self.config.manual_mlp_learning_rate)
        self.mlp_epochs_spin = QSpinBox()
        self.mlp_epochs_spin.setRange(1, 1000)
        self.mlp_epochs_spin.setValue(self.config.manual_mlp_epochs)
        self.mlp_batch_spin = QSpinBox()
        self.mlp_batch_spin.setRange(1, 512)
        self.mlp_batch_spin.setValue(self.config.manual_mlp_batch_size)
        params_layout.addRow("Режим параметров", self.param_mode_combo)
        params_layout.addRow("SVM C", self.svm_c_spin)
        params_layout.addRow("SVM gamma", self.svm_gamma_combo)
        params_layout.addRow("SVM kernel", self.svm_kernel_combo)
        params_layout.addRow("Random Forest n_estimators", self.rf_n_spin)
        params_layout.addRow("Random Forest max_depth (0 = None)", self.rf_depth_spin)
        params_layout.addRow("Random Forest min_samples_split", self.rf_split_spin)
        params_layout.addRow("Random Forest min_samples_leaf", self.rf_leaf_spin)
        params_layout.addRow("MLP hidden layers", self.mlp_hidden_edit)
        params_layout.addRow("MLP learning_rate", self.mlp_lr_spin)
        params_layout.addRow("MLP epochs", self.mlp_epochs_spin)
        params_layout.addRow("MLP batch_size", self.mlp_batch_spin)
        layout.addWidget(params_group)

        self.training_progress = QProgressBar()
        layout.addWidget(self.training_progress)
        start_training_button = QPushButton("Запустить выбранное обучение")
        start_training_button.clicked.connect(self._start_training)
        layout.addWidget(start_training_button)
        self.training_results = QTableWidget(0, 6)
        self.training_results.setHorizontalHeaderLabels(
            ["Модель", "Accuracy", "Balanced acc.", "Macro F1", "Weighted F1", "CV mean"]
        )
        self.training_results.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.training_results)
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        layout.addWidget(self.training_log)

        publish_group = QGroupBox("Публикация модели на сервере")
        publish_layout = QFormLayout(publish_group)
        self.publish_model_path_edit = QLineEdit()
        self.publish_model_path_edit.setReadOnly(True)
        self.publish_version_edit = QLineEdit(f"model-{timestamp()}")
        self.publish_name_edit = QLineEdit("VoiceEmotionApp model")
        self.publish_activate_check = QCheckBox("Сразу назначить модель активной для клиентов")
        self.publish_activate_check.setChecked(True)
        publish_buttons = QHBoxLayout()
        self.btn_publish_model = QPushButton("Опубликовать текущую модель")
        self.btn_publish_model.clicked.connect(self._publish_current_model)
        self.btn_publish_model.setEnabled(False)
        btn_refresh_server_models = QPushButton("Обновить список серверных моделей")
        btn_refresh_server_models.clicked.connect(self._refresh_server_models)
        publish_buttons.addWidget(self.btn_publish_model)
        publish_buttons.addWidget(btn_refresh_server_models)
        publish_layout.addRow("Файл модели", self.publish_model_path_edit)
        publish_layout.addRow("Версия", self.publish_version_edit)
        publish_layout.addRow("Название", self.publish_name_edit)
        publish_layout.addRow(self.publish_activate_check)
        publish_layout.addRow(publish_buttons)
        layout.addWidget(publish_group)

        history_group = QGroupBox("Последние обучения")
        history_layout = QVBoxLayout(history_group)
        hist_buttons = QHBoxLayout()
        btn_refresh = QPushButton("Обновить историю")
        btn_refresh.clicked.connect(self._refresh_history)
        btn_csv = QPushButton("Экспорт истории в CSV")
        btn_csv.clicked.connect(lambda: self._export_history("csv"))
        btn_txt = QPushButton("Экспорт истории в TXT")
        btn_txt.clicked.connect(lambda: self._export_history("txt"))
        hist_buttons.addWidget(btn_refresh)
        hist_buttons.addWidget(btn_csv)
        hist_buttons.addWidget(btn_txt)
        history_layout.addLayout(hist_buttons)
        self.history_table = QTableWidget(0, 7)
        self.history_table.setHorizontalHeaderLabels(["Дата", "Датасет", "Лучшая модель", "Macro F1", "Balanced accuracy", "Модель", "Отчёт"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        history_layout.addWidget(self.history_table)
        layout.addWidget(history_group)
        self.tabs.addTab(tab, "Обучение")

    def _collect_training_options(self):
        selected = [name for name, cb in self.train_model_checks.items() if cb.isChecked()]
        if not selected:
            raise ValueError("Выберите хотя бы одну модель для обучения.")
        hidden_sizes = tuple(
            int(size_text.strip())
            for size_text in self.mlp_hidden_edit.text().split(",")
            if size_text.strip()
        )
        if not hidden_sizes:
            raise ValueError("Для MLP нужно указать хотя бы один скрытый слой, например 128,64.")
        manual_params = {
            "svm": {"C": self.svm_c_spin.value(), "gamma": self.svm_gamma_combo.currentText(), "kernel": self.svm_kernel_combo.currentText()},
            "random_forest": {
                "n_estimators": self.rf_n_spin.value(),
                "max_depth": self.rf_depth_spin.value(),
                "min_samples_split": self.rf_split_spin.value(),
                "min_samples_leaf": self.rf_leaf_spin.value(),
            },
            "mlp": {
                "hidden_sizes": hidden_sizes,
                "lr": self.mlp_lr_spin.value(),
                "epochs": self.mlp_epochs_spin.value(),
                "batch_size": self.mlp_batch_spin.value(),
            },
        }
        return selected, self.param_mode_combo.currentText(), manual_params

    def _start_training(self) -> None:
        if not self.dataset_result:
            self._show_error("Нет датасета", "Сначала загрузите и проверьте CSV-датасет.")
            return
        try:
            selected, parameter_mode, manual_params = self._collect_training_options()
        except Exception as exc:
            self._show_error("Параметры обучения", str(exc))
            return
        self._save_runtime_settings(show_message=False)
        worker = TrainingWorker(self.dataset_result, self.config, self.project_paths, selected_models=selected, parameter_mode=parameter_mode, manual_params=manual_params)
        self._start_worker(worker, self._on_training_finished, self.training_progress, self.training_log)
        self.tabs.setCurrentIndex(3)

    def _fill_training_results_from_metrics(self, metrics: dict) -> None:
        self.training_results.setRowCount(0)
        for row, (name, evaluation) in enumerate(metrics.items()):
            self.training_results.insertRow(row)
            values = [
                name,
                evaluation.get("accuracy"),
                evaluation.get("balanced_accuracy"),
                evaluation.get("macro_f1"),
                evaluation.get("weighted_f1"),
                evaluation.get("cv_mean"),
            ]
            for col, value in enumerate(values):
                if isinstance(value, float):
                    text = f"{value:.4f}"
                elif value is None:
                    text = "—"
                else:
                    text = str(value)
                self.training_results.setItem(row, col, QTableWidgetItem(text))

    def _fill_testing_from_metrics(
        self, metrics: dict, best_model_name: str
    ) -> None:
        text = []
        for name, evaluation in metrics.items():
            text.append(
                f"{name}\n"
                f"Accuracy: {evaluation.get('accuracy', 0):.4f}\n"
                f"Balanced accuracy: {evaluation.get('balanced_accuracy', 0):.4f}\n"
                f"Precision macro: {evaluation.get('precision_macro', 0):.4f}\n"
                f"Recall macro: {evaluation.get('recall_macro', 0):.4f}\n"
                f"Macro F1: {evaluation.get('macro_f1', 0):.4f}\n"
                f"Weighted F1: {evaluation.get('weighted_f1', 0):.4f}\n"
                f"CV mean: {evaluation.get('cv_mean') if evaluation.get('cv_mean') is not None else '—'}\n"
                f"Overfit gap: {evaluation.get('overfit_gap', 0):.4f}\n"
                f"Среднее время распознавания: {evaluation.get('avg_inference_time_sec', 0):.6f} сек.\n"
                f"Параметры: {json.dumps(evaluation.get('best_params', {}), ensure_ascii=False, default=str)}\n"
            )
            for warning in evaluation.get("warnings", []) or []:
                text.append("Предупреждение: " + warning)
        if text:
            self.test_metrics.setText("\n".join(text))
        best_evaluation = metrics.get(best_model_name)
        if best_evaluation:
            confusion_matrix_values = best_evaluation.get("confusion_matrix") or []
            class_labels = list((best_evaluation.get("per_class_f1") or {}).keys())
            self.confusion_table.setRowCount(len(confusion_matrix_values))
            self.confusion_table.setColumnCount(
                len(confusion_matrix_values[0]) if confusion_matrix_values else 0
            )
            if class_labels:
                self.confusion_table.setHorizontalHeaderLabels(class_labels)
                self.confusion_table.setVerticalHeaderLabels(class_labels)
            for row_index, row_values in enumerate(confusion_matrix_values):
                for column_index, cell_value in enumerate(row_values):
                    self.confusion_table.setItem(row_index, column_index, QTableWidgetItem(str(cell_value)))

    def _on_training_finished(self, result: dict) -> None:
        if result.get("skipped"):
            self.current_model_path = result.get("model_path")
            if hasattr(self, "model_path_edit"):
                self.model_path_edit.setText(self.current_model_path or "")
                self._update_recognition_model_choices()
            self.training_log.append("Повторное обучение пропущено: " + result.get("reason", ""))
            self.training_log.append(f"Используется ранее сохранённая модель: {result.get('model_path', '')}")
            self.training_log.append(f"Отчёт: {result.get('report_path', '')}")
            metrics = result.get("metrics") or {}
            self._fill_training_results_from_metrics(metrics)
            self._fill_testing_from_metrics(
                metrics, str(result.get("best_model_name") or "")
            )
            self._prepare_publish_fields(self.current_model_path)
            self._refresh_history()
            self._refresh_home()
            return
        self.current_model_path = result["model_path"]
        if hasattr(self, "model_path_edit"):
            self.model_path_edit.setText(self.current_model_path)
            self._update_recognition_model_choices()
        evaluations = result["evaluations"]
        self.training_results.setRowCount(0)
        for row, (name, evaluation) in enumerate(evaluations.items()):
            self.training_results.insertRow(row)
            values = [
                name,
                evaluation.accuracy,
                evaluation.balanced_accuracy,
                evaluation.macro_f1,
                evaluation.weighted_f1,
                evaluation.cv_mean,
            ]
            for col, value in enumerate(values):
                if isinstance(value, float):
                    text = f"{value:.4f}"
                elif value is None:
                    text = "—"
                else:
                    text = str(value)
                self.training_results.setItem(row, col, QTableWidgetItem(text))
        self.training_log.append(f"Модель сохранена: {result['model_path']}")
        self._prepare_publish_fields(self.current_model_path)
        self.training_log.append(f"Отчёт сохранён: {result['report_path']}")
        self._fill_testing(evaluations, result["bundle"].best_model_name)
        self._refresh_history()
        self._refresh_home()

    def _prepare_publish_fields(self, model_path: str | None) -> None:
        path = str(model_path or "").strip()
        if not hasattr(self, "publish_model_path_edit"):
            return
        self.publish_model_path_edit.setText(path)
        available = bool(path and Path(path).is_file())
        self.btn_publish_model.setEnabled(available)
        if available:
            self.publish_version_edit.setText(f"model-{timestamp()}")
            self.publish_name_edit.setText(f"VoiceEmotionApp {Path(path).stem}")

    def _publish_current_model(self) -> None:
        model_path = self.publish_model_path_edit.text().strip() if hasattr(self, "publish_model_path_edit") else ""
        if not model_path or not Path(model_path).is_file():
            self._show_error("Публикация модели", "Сначала обучите модель или выберите существующий файл модели.")
            return
        self._save_server_settings(show_message=False)
        version = self.publish_version_edit.text().strip()
        display_name = self.publish_name_edit.text().strip()
        activate = self.publish_activate_check.isChecked()
        self.training_log.append(f"Публикация модели {version} на сервере...")
        self.btn_publish_model.setEnabled(False)
        worker = FunctionWorker(
            publish_model,
            self.config.server_url,
            self.config.server_dir,
            model_path,
            version,
            display_name,
            activate,
        )
        self._start_worker(worker, self._on_model_published, None, self.training_log, self._on_model_publish_failed)

    def _on_model_publish_failed(self, message: str) -> None:
        if hasattr(self, "btn_publish_model"):
            self.btn_publish_model.setEnabled(bool(self.current_model_path and Path(self.current_model_path).is_file()))
        if hasattr(self, "training_log"):
            self.training_log.append("Публикация модели не выполнена: " + message)

    def _on_model_published(self, result: dict) -> None:
        if hasattr(self, "btn_publish_model"):
            self.btn_publish_model.setEnabled(True)
        version = str(result.get("version", self.publish_version_edit.text().strip()))
        sha256 = str(result.get("sha256", ""))
        activated = bool(result.get("activation"))
        self.training_log.append(f"Модель опубликована: {version}")
        if sha256:
            self.training_log.append(f"SHA-256: {sha256}")
        if activated:
            self.training_log.append("Модель назначена активной для клиентов.")
        self._refresh_server_models()
        self._show_info("Публикация модели", f"Модель {version} опубликована на сервере." + ("\nОна назначена активной." if activated else ""))

    def _refresh_history(self) -> None:
        if not hasattr(self, "history_table"):
            return
        experiment_database = ExperimentDB(self.project_paths.database / "experiments.sqlite")
        rows = experiment_database.last_experiments(20)
        self.history_table.setRowCount(0)
        for row_index, row in enumerate(rows):
            metrics = json.loads(row.get("metrics_json") or "{}")
            best_name = str(row.get("best_model_name") or "—")
            macro = "—"
            balanced_accuracy = "—"
            if best_name in metrics:
                best_evaluation = metrics[best_name]
                macro = f"{best_evaluation.get('macro_f1', 0):.4f}"
                balanced_accuracy = f"{best_evaluation.get('balanced_accuracy', 0):.4f}"
            values = [row.get("created_at", ""), Path(row.get("dataset_path", "")).name, best_name, macro, balanced_accuracy, row.get("model_path", ""), row.get("report_path", "")]
            self.history_table.insertRow(row_index)
            for column_index, value in enumerate(values):
                self.history_table.setItem(row_index, column_index, QTableWidgetItem(str(value)))

    def _export_history(self, export_format: str) -> None:
        experiment_database = ExperimentDB(self.project_paths.database / "experiments.sqlite")
        output_path = self.project_paths.exports / (
            f"experiments_history_{timestamp()}.{export_format}"
        )
        if export_format == "csv":
            path = experiment_database.export_csv(output_path)
        else:
            path = experiment_database.export_txt(output_path)
        self._show_info("Экспорт истории", f"История экспериментов сохранена:\n{path}")
