"""Диалоги сопоставления эмоций и восстановления данных."""

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..constants import AGE_GROUPS, EMOTIONS, EMOTION_RU, GENDERS, LANGUAGES
from ..dataset import EXCLUDE_MARK

class EmotionMappingDialog(QDialog):

    def __init__(self, label_information, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Сопоставление эмоций датасета")
        self.resize(720, 420)
        layout = QVBoxLayout(self)
        hint = QLabel("Проверьте найденные в CSV метки эмоций. Для каждой метки выберите поддерживаемый класс или исключение из обработки.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Исходная метка", "Количество", "Сопоставить с"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)
        for row, label_info in enumerate(label_information):
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(label_info.label))
            self.table.setItem(row, 1, QTableWidgetItem(str(label_info.count)))
            combo = QComboBox()
            combo.addItem("исключить", EXCLUDE_MARK)
            for emotion in EMOTIONS:
                combo.addItem(f"{emotion} — {EMOTION_RU[emotion]}", emotion)
            if label_info.suggested:
                combo.setCurrentIndex(EMOTIONS.index(label_info.suggested) + 1)
            self.table.setCellWidget(row, 2, combo)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def mapping(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for row in range(self.table.rowCount()):
            label_item = self.table.item(row, 0)
            combo = self.table.cellWidget(row, 2)
            if label_item and isinstance(combo, QComboBox):
                result[label_item.text()] = combo.currentData()
        return result

class RecoveryMetadataDialog(QDialog):

    def __init__(self, speaker_counts: dict[str, int], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Восстановление CSV пользовательского датасета")
        self.resize(820, 420)
        layout = QVBoxLayout(self)
        hint = QLabel(
            "Программа нашла аудиозаписи в datasets/user_dataset и может заново создать user_dataset.csv. "
            "Укажите сведения о записывающих: пол, возрастную группу и язык. Эти данные будут записаны в CSV."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Диктор", "Найдено записей", "Пол", "Возрастная группа", "Язык"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

        for row, (speaker_id, count) in enumerate(sorted(speaker_counts.items())):
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(speaker_id))
            self.table.setItem(row, 1, QTableWidgetItem(str(count)))

            gender_combo = QComboBox()
            gender_combo.addItems(GENDERS)
            self.table.setCellWidget(row, 2, gender_combo)

            age_combo = QComboBox()
            age_combo.addItems(AGE_GROUPS)
            self.table.setCellWidget(row, 3, age_combo)

            lang_combo = QComboBox()
            lang_combo.addItems(LANGUAGES)
            lang_combo.setCurrentText("ru")
            self.table.setCellWidget(row, 4, lang_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self) -> None:
        for row in range(self.table.rowCount()):
            speaker_item = self.table.item(row, 0)
            speaker_id = speaker_item.text() if speaker_item else f"строка {row + 1}"
            gender_combo = self.table.cellWidget(row, 2)
            age_combo = self.table.cellWidget(row, 3)
            lang_combo = self.table.cellWidget(row, 4)
            gender = gender_combo.currentText().strip() if isinstance(gender_combo, QComboBox) else ""
            age_group = age_combo.currentText().strip() if isinstance(age_combo, QComboBox) else ""
            language = lang_combo.currentText().strip() if isinstance(lang_combo, QComboBox) else ""
            if not gender or gender == "не указано":
                QMessageBox.warning(self, "Не заполнены данные", f"Укажите пол для диктора {speaker_id}.")
                return
            if not age_group:
                QMessageBox.warning(self, "Не заполнены данные", f"Укажите возрастную группу для диктора {speaker_id}.")
                return
            if not language:
                QMessageBox.warning(self, "Не заполнены данные", f"Укажите язык для диктора {speaker_id}.")
                return
        super().accept()

    def metadata(self) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for row in range(self.table.rowCount()):
            speaker_item = self.table.item(row, 0)
            if not speaker_item:
                continue
            gender_combo = self.table.cellWidget(row, 2)
            age_combo = self.table.cellWidget(row, 3)
            lang_combo = self.table.cellWidget(row, 4)
            speaker_id = speaker_item.text()
            result[speaker_id] = {
                "gender": gender_combo.currentText().strip() if isinstance(gender_combo, QComboBox) else "",
                "age_group": age_combo.currentText().strip() if isinstance(age_combo, QComboBox) else "",
                "language": lang_combo.currentText().strip() if isinstance(lang_combo, QComboBox) else "ru",
            }
        return result
