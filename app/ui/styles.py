"""Стили светлой и тёмной темы приложения."""

DARK_STYLESHEET = """
QWidget { background: #202124; color: #f1f3f4; font-size: 13px; }
QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget {
    background: #2b2c30; color: #f1f3f4; border: 1px solid #555; padding: 4px;
}
QPushButton { background: #3c4043; color: #f1f3f4; border: 1px solid #666; padding: 7px 12px; border-radius: 4px; }
QPushButton:hover { background: #4a4d51; }
QGroupBox { border: 1px solid #555; margin-top: 12px; padding: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QHeaderView::section { background: #303134; color: #f1f3f4; padding: 4px; border: 1px solid #555; }
"""

LIGHT_STYLESHEET = """
QWidget { background: #fafafa; color: #202124; font-size: 13px; }
QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget {
    background: #ffffff; color: #202124; border: 1px solid #c9c9c9; padding: 4px;
}
QPushButton { background: #f0f0f0; color: #202124; border: 1px solid #bdbdbd; padding: 7px 12px; border-radius: 4px; }
QPushButton:hover { background: #e3e3e3; }
QGroupBox { border: 1px solid #c9c9c9; margin-top: 12px; padding: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QHeaderView::section { background: #eeeeee; color: #202124; padding: 4px; border: 1px solid #c9c9c9; }
"""
