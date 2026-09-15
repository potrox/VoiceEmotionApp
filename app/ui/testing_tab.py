"""Вкладка проверки обученной модели на размеченных данных."""

import json

from PySide6.QtWidgets import (
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

class TestingTabMixin:

    def _build_testing_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.test_metrics = QTextEdit()
        self.test_metrics.setReadOnly(True)
        layout.addWidget(self.test_metrics)
        self.confusion_table = QTableWidget(0, 0)
        layout.addWidget(self.confusion_table)
        self.tabs.addTab(tab, "Тестирование")

    def _fill_testing(self, evaluations: dict, best_model_name: str) -> None:
        text = []
        for name, evaluation in evaluations.items():
            text.append(
                f"{name}\n"
                f"Accuracy: {evaluation.accuracy:.4f}\n"
                f"Balanced accuracy: {evaluation.balanced_accuracy:.4f}\n"
                f"Precision macro: {evaluation.precision_macro:.4f}\n"
                f"Recall macro: {evaluation.recall_macro:.4f}\n"
                f"Macro F1: {evaluation.macro_f1:.4f}\n"
                f"Weighted F1: {evaluation.weighted_f1:.4f}\n"
                f"CV mean: {evaluation.cv_mean if evaluation.cv_mean is not None else '—'}\n"
                f"Overfit gap: {evaluation.overfit_gap:.4f}\n"
                f"Среднее время распознавания: {evaluation.avg_inference_time_sec:.6f} сек.\n"
                f"Параметры: {json.dumps(evaluation.best_params, ensure_ascii=False, default=str)}\n"
            )
            if evaluation.warnings:
                text.extend(["Предупреждение: " + warning for warning in evaluation.warnings])
        self.test_metrics.setText("\n".join(text))
        best_evaluation = evaluations.get(best_model_name)
        if best_evaluation:
            confusion_matrix_values = best_evaluation.confusion_matrix
            self.confusion_table.setRowCount(len(confusion_matrix_values))
            self.confusion_table.setColumnCount(len(confusion_matrix_values))
            class_labels = list(best_evaluation.per_class_f1.keys())
            self.confusion_table.setHorizontalHeaderLabels(class_labels)
            self.confusion_table.setVerticalHeaderLabels(class_labels)
            for row_index, row in enumerate(confusion_matrix_values):
                for column_index, cell_value in enumerate(row):
                    self.confusion_table.setItem(row_index, column_index, QTableWidgetItem(str(cell_value)))
