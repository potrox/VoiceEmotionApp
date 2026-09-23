from __future__ import annotations

from typing import Dict
import json

from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget


class TestingTabMixin:
    def _build_testing_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.test_metrics = QTextEdit(); self.test_metrics.setReadOnly(True)
        layout.addWidget(self.test_metrics)
        self.confusion_table = QTableWidget(0, 0)
        layout.addWidget(self.confusion_table)
        self.tabs.addTab(tab, "Тестирование")

    def _fill_testing(self, evaluations: Dict) -> None:
        text = []
        best_eval = None
        for name, ev in evaluations.items():
            intervals = ev.metric_intervals or {}
            text.append(
                f"{name}\n"
                f"Accuracy: {ev.accuracy:.4f}\nBalanced accuracy: {ev.balanced_accuracy:.4f}\n"
                f"Precision macro: {ev.precision_macro:.4f}\nRecall macro: {ev.recall_macro:.4f}\n"
                f"Macro F1: {ev.macro_f1:.4f}\nWeighted F1: {ev.weighted_f1:.4f}\n"
                f"Macro F1 95% CI: {intervals.get('macro_f1_95ci', '—')}\n"
                f"Balanced accuracy 95% CI: {intervals.get('balanced_accuracy_95ci', '—')}\n"
                f"CV mean: {ev.cv_mean if ev.cv_mean is not None else '—'}\n"
                f"Overfit gap: {ev.overfit_gap:.4f}\nСреднее время распознавания: {ev.avg_inference_time_sec:.6f} сек.\n"
                f"Параметры: {json.dumps(ev.best_params, ensure_ascii=False, default=str)}\n"
            )
            if ev.warnings:
                text.extend(["Предупреждение: " + w for w in ev.warnings])
            if best_eval is None or ev.complex_score > best_eval.complex_score:
                best_eval = ev
        self.test_metrics.setText("\n".join(text))
        if best_eval:
            cm = best_eval.confusion_matrix
            self.confusion_table.setRowCount(len(cm)); self.confusion_table.setColumnCount(len(cm))
            labels = list(best_eval.per_class_f1.keys())
            self.confusion_table.setHorizontalHeaderLabels(labels)
            self.confusion_table.setVerticalHeaderLabels(labels)
            for i, row in enumerate(cm):
                for j, val in enumerate(row):
                    self.confusion_table.setItem(i, j, QTableWidgetItem(str(val)))
