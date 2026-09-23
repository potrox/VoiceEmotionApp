from __future__ import annotations

from pathlib import Path
from typing import Dict
import csv
import json

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .gui_shared import MODEL_OPTIONS
from .storage import ExperimentDB, timestamp
from .workers import TrainingWorker


class TrainingTabMixin:
    def _build_training_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        model_group = QGroupBox("Выбор моделей для обучения")
        model_layout = QHBoxLayout(model_group)
        self.train_model_checks: Dict[str, QCheckBox] = {}
        for name in MODEL_OPTIONS:
            cb = QCheckBox(name)
            cb.setChecked(name in self.cfg.training_models)
            self.train_model_checks[name] = cb
            model_layout.addWidget(cb)
        layout.addWidget(model_group)

        params_group = QGroupBox("Протокол обучения")
        params_layout = QFormLayout(params_group)
        self.param_mode_combo = QComboBox(); self.param_mode_combo.addItems(["auto"])
        params_layout.addRow("Подбор параметров", self.param_mode_combo)
        protocol_label = QLabel(
            "5-fold grouped CV по дикторам; медианная импутация и отбор признаков "
            "внутри CV; настройка решающего правила только по OOF-прогнозам train."
        )
        protocol_label.setWordWrap(True)
        params_layout.addRow("Проверка", protocol_label)
        layout.addWidget(params_group)

        self.training_progress = QProgressBar()
        layout.addWidget(self.training_progress)
        btn = QPushButton("Запустить выбранное обучение")
        btn.clicked.connect(self._start_training)
        layout.addWidget(btn)
        self.training_results = QTableWidget(0, 9)
        self.training_results.setHorizontalHeaderLabels([
            "Модель", "Balanced acc.", "Macro F1", "Raw F1", "ROC-AUC",
            "PR-AUC", "CV/OOF F1", "OOF gain", "Train-test gap"
        ])
        self.training_results.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.training_results)
        self.training_log = QTextEdit(); self.training_log.setReadOnly(True)
        layout.addWidget(self.training_log)

        history_group = QGroupBox("Последние обучения")
        history_layout = QVBoxLayout(history_group)
        hist_buttons = QHBoxLayout()
        btn_refresh = QPushButton("Обновить историю")
        btn_refresh.clicked.connect(self._refresh_history)
        btn_csv = QPushButton("Экспорт истории в CSV")
        btn_csv.clicked.connect(lambda: self._export_history("csv"))
        btn_txt = QPushButton("Экспорт истории в TXT")
        btn_txt.clicked.connect(lambda: self._export_history("txt"))
        hist_buttons.addWidget(btn_refresh); hist_buttons.addWidget(btn_csv); hist_buttons.addWidget(btn_txt)
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
        return selected, "auto", {}

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
        worker = TrainingWorker(self.dataset_result, self.cfg, self.paths, selected_models=selected, parameter_mode=parameter_mode, manual_params=manual_params)
        self._start_worker(worker, self._on_training_finished, self.training_progress, self.training_log)
        self.tabs.setCurrentIndex(3)

    def _fill_training_results_from_metrics(self, metrics: Dict) -> None:
        self.training_results.setRowCount(0)
        for row, (name, ev) in enumerate(metrics.items()):
            self.training_results.insertRow(row)
            values = [
                name,
                ev.get("balanced_accuracy"),
                ev.get("macro_f1"),
                ev.get("raw_test_macro_f1"),
                ev.get("roc_auc_macro"),
                ev.get("average_precision_macro"),
                ev.get("cv_mean"),
                ev.get("threshold_cv_gain"),
                ev.get("overfit_gap"),
            ]
            for col, value in enumerate(values):
                if isinstance(value, float):
                    text = f"{value:.4f}"
                elif value is None:
                    text = "—"
                else:
                    text = str(value)
                self.training_results.setItem(row, col, QTableWidgetItem(text))

    def _fill_testing_from_metrics(self, metrics: Dict) -> None:
        text = []
        best_name = None
        best_ev = None
        for name, ev in metrics.items():
            intervals = ev.get("metric_intervals") or {}
            text.append(
                f"{name}\n"
                f"Accuracy: {ev.get('accuracy', 0):.4f}\nBalanced accuracy: {ev.get('balanced_accuracy', 0):.4f}\n"
                f"Precision macro: {ev.get('precision_macro', 0):.4f}\nRecall macro: {ev.get('recall_macro', 0):.4f}\n"
                f"Macro F1: {ev.get('macro_f1', 0):.4f}\nWeighted F1: {ev.get('weighted_f1', 0):.4f}\n"
                f"Raw macro F1 до OOF-смещений: {ev.get('raw_test_macro_f1', 0):.4f}\n"
                f"ROC-AUC macro: {ev.get('roc_auc_macro', 0):.4f}\n"
                f"PR-AUC macro: {ev.get('average_precision_macro', 0):.4f}\n"
                f"Macro F1 95% CI: {intervals.get('macro_f1_95ci', '—')}\n"
                f"Balanced accuracy 95% CI: {intervals.get('balanced_accuracy_95ci', '—')}\n"
                f"Grouped OOF macro F1: {ev.get('cv_mean') if ev.get('cv_mean') is not None else '—'}\n"
                f"OOF threshold gain: {ev.get('threshold_cv_gain', 0):.4f}\n"
                f"Overfit gap: {ev.get('overfit_gap', 0):.4f}\n"
                f"Среднее время распознавания: {ev.get('avg_inference_time_sec', 0):.6f} сек.\n"
                f"Параметры: {json.dumps(ev.get('best_params', {}), ensure_ascii=False, default=str)}\n"
            )
            for warning in ev.get("warnings", []) or []:
                text.append("Предупреждение: " + warning)
            if best_ev is None or ev.get("complex_score", 0) > best_ev.get("complex_score", 0):
                best_name, best_ev = name, ev
        if text:
            self.test_metrics.setText("\n".join(text))
        if best_ev:
            cm = best_ev.get("confusion_matrix") or []
            labels = list((best_ev.get("per_class_f1") or {}).keys())
            self.confusion_table.setRowCount(len(cm)); self.confusion_table.setColumnCount(len(cm[0]) if cm else 0)
            if labels:
                self.confusion_table.setHorizontalHeaderLabels(labels)
                self.confusion_table.setVerticalHeaderLabels(labels)
            for i, row_values in enumerate(cm):
                for j, val in enumerate(row_values):
                    self.confusion_table.setItem(i, j, QTableWidgetItem(str(val)))

    def _on_training_finished(self, result: Dict) -> None:
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
            self._fill_testing_from_metrics(metrics)
            self._refresh_history()
            self._refresh_home()
            return
        self.current_model_path = result["model_path"]
        if hasattr(self, "model_path_edit"):
            self.model_path_edit.setText(self.current_model_path)
            self._update_recognition_model_choices()
        evaluations = result["evaluations"]
        self.training_results.setRowCount(0)
        for row, (name, ev) in enumerate(evaluations.items()):
            self.training_results.insertRow(row)
            values = [name, ev.accuracy, ev.balanced_accuracy, ev.macro_f1, ev.weighted_f1, ev.cv_mean, ev.complex_score]
            for col, value in enumerate(values):
                if isinstance(value, float):
                    text = f"{value:.4f}"
                elif value is None:
                    text = "—"
                else:
                    text = str(value)
                self.training_results.setItem(row, col, QTableWidgetItem(text))
        self.training_log.append(f"Модель сохранена: {result['model_path']}")
        self.training_log.append(f"Отчёт сохранён: {result['report_path']}")
        self._fill_testing(evaluations)
        self._refresh_history()
        self._refresh_home()

    def _refresh_history(self) -> None:
        if not hasattr(self, "history_table"):
            return
        db = ExperimentDB(self.paths.database / "experiments.sqlite")
        rows = db.last_experiments(20)
        self.history_table.setRowCount(0)
        for r, row in enumerate(rows):
            metrics = json.loads(row.get("metrics_json") or "{}")
            best_name = "—"
            macro = "—"
            bal = "—"
            if metrics:
                best_name, best_ev = max(metrics.items(), key=lambda item: item[1].get("complex_score", 0))
                macro = f"{best_ev.get('macro_f1', 0):.4f}"
                bal = f"{best_ev.get('balanced_accuracy', 0):.4f}"
            values = [row.get("created_at", ""), Path(row.get("dataset_path", "")).name, best_name, macro, bal, row.get("model_path", ""), row.get("report_path", "")]
            self.history_table.insertRow(r)
            for c, value in enumerate(values):
                self.history_table.setItem(r, c, QTableWidgetItem(str(value)))

    def _export_history(self, fmt: str) -> None:
        db = ExperimentDB(self.paths.database / "experiments.sqlite")
        out = self.paths.exports / f"experiments_history_{timestamp()}.{fmt}"
        if fmt == "csv":
            path = db.export_csv(out)
        else:
            path = db.export_txt(out)
        self._show_info("Экспорт истории", f"История экспериментов сохранена:\n{path}")
