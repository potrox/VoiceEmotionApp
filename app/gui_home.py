from __future__ import annotations

import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget

from .constants import EMOTION_RU
from .storage import ExperimentDB


class HomeTabMixin:
    def _build_home_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        title = QLabel("VoiceEmotionApp")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 28px; font-weight: 600;")
        layout.addWidget(title)
        self.home_status = QTextEdit()
        self.home_status.setReadOnly(True)
        layout.addWidget(self.home_status)
        buttons = QHBoxLayout()
        btn_dataset = QPushButton("Загрузить CSV-датасет")
        btn_dataset.clicked.connect(self._select_dataset_csv)
        btn_train = QPushButton("Запустить обучение")
        btn_train.clicked.connect(self._start_training)
        btn_recognize = QPushButton("Распознать файл")
        btn_recognize.clicked.connect(self._recognize_file_dialog)
        buttons.addWidget(btn_dataset)
        buttons.addWidget(btn_train)
        buttons.addWidget(btn_recognize)
        layout.addLayout(buttons)
        self.tabs.addTab(tab, "Главная")

    def _refresh_home(self) -> None:
        lines = []
        if self.dataset_result:
            lines.append(f"Датасет: загружен ({self.dataset_result.records_count} записей)")
            lines.append("Распределение: " + ", ".join(f"{EMOTION_RU.get(k, k)}: {v}" for k, v in self.dataset_result.distribution.items()))
        else:
            lines.append("Датасет: не загружен")
        lines.append(f"Модель: {self.current_model_path or 'не загружена'}")
        try:
            db = ExperimentDB(self.paths.database / "experiments.sqlite")
            last = db.last_experiments(1)
            if last:
                row = last[0]
                metrics = json.loads(row.get("metrics_json") or "{}")
                lines.append(f"Последнее обучение: {row.get('created_at')}")
                lines.append(f"Последняя сохранённая модель: {row.get('model_path', '')}")
                if metrics:
                    best_name, best_ev = max(metrics.items(), key=lambda item: item[1].get("complex_score", 0))
                    lines.append(f"Последняя лучшая модель: {best_name}")
                    lines.append(f"Macro F1: {best_ev.get('macro_f1', 0):.3f}; balanced accuracy: {best_ev.get('balanced_accuracy', 0):.3f}")
        except Exception:
            pass
        unique_lines = []
        seen_lines = set()
        for line in lines:
            if line not in seen_lines:
                unique_lines.append(line)
                seen_lines.add(line)
        self.home_status.setText("\n".join(unique_lines))
