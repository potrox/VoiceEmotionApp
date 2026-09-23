"""Вкладка управления сервером и опубликованными моделями."""

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .server_integration import (
    activate_server_model,
    create_backup,
    create_registration_code,
    list_server_devices,
    list_server_models,
    open_directory,
    open_url,
    restart_server,
    run_migrations,
    run_server_test,
    server_status,
    service_logs,
    start_server,
    stop_server,
    validate_server_directory,
    verify_server_environment,
)
from .workers import FunctionWorker

class ServerTabMixin:

    def _build_server_tab(self) -> None:
        tab = QWidget()
        self.server_tab = tab
        layout = QVBoxLayout(tab)

        connection_group = QGroupBox("Подключение и расположение сервера")
        connection_layout = QFormLayout(connection_group)
        self.server_url_edit = QLineEdit(self.cfg.server_url)
        self.server_dir_edit = QLineEdit(self.cfg.server_dir)
        server_dir_row = QHBoxLayout()
        server_dir_row.addWidget(self.server_dir_edit)
        btn_server_dir = QPushButton("Выбрать папку")
        btn_server_dir.clicked.connect(self._select_server_directory)
        server_dir_row.addWidget(btn_server_dir)
        connection_layout.addRow("Адрес API", self.server_url_edit)
        connection_layout.addRow("Папка сервера", server_dir_row)
        connection_buttons = QHBoxLayout()
        btn_save_server = QPushButton("Сохранить настройки")
        btn_save_server.clicked.connect(self._save_server_settings)
        btn_verify_server = QPushButton("Проверить папку и .env")
        btn_verify_server.clicked.connect(self._verify_server_environment)
        btn_open_server = QPushButton("Открыть папку сервера")
        btn_open_server.clicked.connect(self._open_server_directory)
        btn_open_docs = QPushButton("Открыть API /docs")
        btn_open_docs.clicked.connect(self._open_server_docs)
        connection_buttons.addWidget(btn_save_server)
        connection_buttons.addWidget(btn_verify_server)
        connection_buttons.addWidget(btn_open_server)
        connection_buttons.addWidget(btn_open_docs)
        connection_layout.addRow(connection_buttons)
        layout.addWidget(connection_group)

        control_group = QGroupBox("Управление Docker-сервером")
        control_layout = QVBoxLayout(control_group)
        row_one = QHBoxLayout()
        btn_start = QPushButton("Запустить сервер")
        btn_start.clicked.connect(lambda: self._run_server_action("Запуск сервера", start_server, self.server_dir_edit.text().strip(), False))
        btn_rebuild = QPushButton("Пересобрать и запустить")
        btn_rebuild.clicked.connect(lambda: self._run_server_action("Пересборка сервера", start_server, self.server_dir_edit.text().strip(), True))
        btn_stop = QPushButton("Остановить сервер")
        btn_stop.clicked.connect(lambda: self._run_server_action("Остановка сервера", stop_server, self.server_dir_edit.text().strip()))
        btn_restart = QPushButton("Перезапустить сервер")
        btn_restart.clicked.connect(lambda: self._run_server_action("Перезапуск сервера", restart_server, self.server_dir_edit.text().strip()))
        row_one.addWidget(btn_start)
        row_one.addWidget(btn_rebuild)
        row_one.addWidget(btn_stop)
        row_one.addWidget(btn_restart)
        control_layout.addLayout(row_one)

        row_two = QHBoxLayout()
        btn_status = QPushButton("Проверить состояние")
        btn_status.clicked.connect(self._refresh_server_status)
        btn_migrations = QPushButton("Выполнить миграции")
        btn_migrations.clicked.connect(lambda: self._run_server_action("Миграции", run_migrations, self.server_dir_edit.text().strip()))
        btn_test = QPushButton("Полный тест сервера")
        btn_test.clicked.connect(lambda: self._run_server_action("Тест сервера", run_server_test, self.server_dir_edit.text().strip()))
        btn_backup = QPushButton("Создать резервную копию")
        btn_backup.clicked.connect(lambda: self._run_server_action("Резервное копирование", create_backup, self.server_dir_edit.text().strip()))
        btn_open_backups = QPushButton("Открыть backups")
        btn_open_backups.clicked.connect(self._open_backups_directory)
        row_two.addWidget(btn_status)
        row_two.addWidget(btn_migrations)
        row_two.addWidget(btn_test)
        row_two.addWidget(btn_backup)
        row_two.addWidget(btn_open_backups)
        control_layout.addLayout(row_two)
        layout.addWidget(control_group)

        registration_group = QGroupBox("Регистрация клиентов")
        registration_layout = QHBoxLayout(registration_group)
        registration_layout.addWidget(QLabel("Срок действия кода, минут"))
        self.registration_minutes_spin = QSpinBox()
        self.registration_minutes_spin.setRange(1, 1440)
        self.registration_minutes_spin.setValue(60)
        registration_layout.addWidget(self.registration_minutes_spin)
        btn_code = QPushButton("Создать одноразовый код")
        btn_code.clicked.connect(self._create_registration_code)
        registration_layout.addWidget(btn_code)
        self.registration_code_edit = QLineEdit()
        self.registration_code_edit.setReadOnly(True)
        self.registration_code_edit.setPlaceholderText("Код появится здесь")
        registration_layout.addWidget(self.registration_code_edit, 1)
        layout.addWidget(registration_group)

        models_group = QGroupBox("Опубликованные модели")
        models_layout = QVBoxLayout(models_group)
        model_buttons = QHBoxLayout()
        btn_models_refresh = QPushButton("Обновить модели")
        btn_models_refresh.clicked.connect(self._refresh_server_models)
        btn_activate = QPushButton("Назначить выбранную модель активной")
        btn_activate.clicked.connect(self._activate_selected_server_model)
        model_buttons.addWidget(btn_models_refresh)
        model_buttons.addWidget(btn_activate)
        models_layout.addLayout(model_buttons)
        self.server_models_table = QTableWidget(0, 6)
        self.server_models_table.setHorizontalHeaderLabels(["Версия", "Название", "Тип", "SHA-256", "Активна", "Создана"])
        self.server_models_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.server_models_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.server_models_table.setSelectionMode(QAbstractItemView.SingleSelection)
        models_layout.addWidget(self.server_models_table)
        layout.addWidget(models_group)

        devices_group = QGroupBox("Подключённые клиентские устройства")
        devices_layout = QVBoxLayout(devices_group)
        btn_devices = QPushButton("Обновить список клиентов")
        btn_devices.clicked.connect(self._refresh_server_devices)
        devices_layout.addWidget(btn_devices)
        self.server_devices_table = QTableWidget(0, 8)
        self.server_devices_table.setHorizontalHeaderLabels(["Устройство", "Пользователь ID", "Онлайн", "Мониторинг", "Версия клиента", "Версия модели", "Последняя связь", "Заблокировано"])
        self.server_devices_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        devices_layout.addWidget(self.server_devices_table)
        layout.addWidget(devices_group)

        logs_group = QGroupBox("Состояние и журналы сервисов")
        logs_layout = QVBoxLayout(logs_group)
        logs_controls = QHBoxLayout()
        self.server_log_service_combo = QComboBox()
        self.server_log_service_combo.addItems(["api", "worker", "scheduler", "db", "backup", "migrate"])
        btn_logs = QPushButton("Показать последние логи")
        btn_logs.clicked.connect(self._show_server_logs)
        btn_clear_logs = QPushButton("Очистить окно")
        btn_clear_logs.clicked.connect(lambda: self.server_log.clear())
        logs_controls.addWidget(QLabel("Сервис"))
        logs_controls.addWidget(self.server_log_service_combo)
        logs_controls.addWidget(btn_logs)
        logs_controls.addWidget(btn_clear_logs)
        logs_layout.addLayout(logs_controls)
        self.server_log = QTextEdit()
        self.server_log.setReadOnly(True)
        logs_layout.addWidget(self.server_log)
        layout.addWidget(logs_group)
        self.tabs.addTab(tab, "Сервер")

    def _select_server_directory(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку VoiceEmotionServer", self.server_dir_edit.text().strip() or self.cfg.server_dir)
        if folder:
            self.server_dir_edit.setText(folder)
            self._save_server_settings(show_message=False)

    def _save_server_settings(self, show_message: bool = True) -> None:
        if hasattr(self, "server_url_edit"):
            self.cfg.server_url = self.server_url_edit.text().strip().rstrip("/") or "http://127.0.0.1:8000"
        if hasattr(self, "server_dir_edit"):
            self.cfg.server_dir = self.server_dir_edit.text().strip() or "server"
        self.cfg.save("config.json")
        if show_message:
            self._show_info("Настройки сервера", "Адрес API и папка сервера сохранены.")

    def _server_action_failed(self, message: str) -> None:
        if hasattr(self, "server_log"):
            self.server_log.append("Ошибка: " + message)

    def _run_server_action(self, title: str, operation, *operation_args) -> None:
        self._save_server_settings(show_message=False)
        if hasattr(self, "server_log"):
            self.server_log.append(f"\n=== {title} ===")
        worker = FunctionWorker(operation, *operation_args)
        def done(result) -> None:
            if hasattr(self, "server_log"):
                self.server_log.append(str(result))
            if title in {"Запуск сервера", "Пересборка сервера", "Перезапуск сервера", "Миграции"}:
                QTimer.singleShot(500, self._refresh_server_status)
        self._start_worker(worker, done, None, self.server_log, self._server_action_failed)

    def _verify_server_environment(self) -> None:
        self._save_server_settings(show_message=False)
        self._run_server_action("Проверка серверной папки", verify_server_environment, self.cfg.server_dir)

    def _refresh_server_status(self) -> None:
        self._save_server_settings(show_message=False)
        self._run_server_action("Состояние сервера", server_status, self.cfg.server_dir, self.cfg.server_url)

    def _show_server_logs(self) -> None:
        self._save_server_settings(show_message=False)
        service = self.server_log_service_combo.currentText()
        self._run_server_action(f"Логи {service}", service_logs, self.cfg.server_dir, service, 300)

    def _open_server_directory(self) -> None:
        try:
            self._save_server_settings(show_message=False)
            open_directory(validate_server_directory(self.cfg.server_dir))
        except Exception as exc:
            self._show_error("Папка сервера", str(exc))

    def _open_backups_directory(self) -> None:
        try:
            self._save_server_settings(show_message=False)
            open_directory(validate_server_directory(self.cfg.server_dir) / "backups")
        except Exception as exc:
            self._show_error("Резервные копии", str(exc))

    def _open_server_docs(self) -> None:
        self._save_server_settings(show_message=False)
        open_url(self.cfg.server_url.rstrip("/") + "/docs")

    def _create_registration_code(self) -> None:
        self._save_server_settings(show_message=False)
        minutes = self.registration_minutes_spin.value()
        worker = FunctionWorker(create_registration_code, self.cfg.server_url, self.cfg.server_dir, minutes)
        def done(result) -> None:
            code = str(result.get("code", ""))
            expires = str(result.get("expires_at", ""))
            self.registration_code_edit.setText(code)
            self.server_log.append(f"Создан одноразовый код: {code}\nДействует до: {expires}")
        self._start_worker(worker, done, None, self.server_log, self._server_action_failed)

    def _refresh_server_models(self) -> None:
        if not hasattr(self, "server_models_table"):
            return
        self._save_server_settings(show_message=False)
        worker = FunctionWorker(list_server_models, self.cfg.server_url, self.cfg.server_dir)
        self._start_worker(worker, self._fill_server_models, None, self.server_log, self._server_action_failed)

    def _fill_server_models(self, models: list[dict]) -> None:
        self.server_models_table.setRowCount(0)
        for row, model in enumerate(models):
            self.server_models_table.insertRow(row)
            values = [
                model.get("version", ""),
                model.get("display_name", ""),
                model.get("model_type", ""),
                model.get("sha256", ""),
                "да" if model.get("is_published") else "нет",
                model.get("created_at", ""),
            ]
            for col, value in enumerate(values):
                self.server_models_table.setItem(row, col, QTableWidgetItem(str(value)))
        if hasattr(self, "server_log"):
            self.server_log.append(f"Получено моделей: {len(models)}")

    def _activate_selected_server_model(self) -> None:
        row = self.server_models_table.currentRow()
        if row < 0:
            self._show_error("Активация модели", "Выберите строку модели в таблице.")
            return
        item = self.server_models_table.item(row, 0)
        version = item.text().strip() if item else ""
        if not version:
            self._show_error("Активация модели", "Не удалось определить версию выбранной модели.")
            return
        answer = QMessageBox.question(
            self,
            "Назначение модели",
            f"Назначить модель {version} активной для всех клиентов?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        worker = FunctionWorker(activate_server_model, self.cfg.server_url, self.cfg.server_dir, version)
        def done(result) -> None:
            self.server_log.append(f"Активная модель: {result.get('active_version', version)}")
            self._refresh_server_models()
        self._start_worker(worker, done, None, self.server_log, self._server_action_failed)

    def _refresh_server_devices(self) -> None:
        self._save_server_settings(show_message=False)
        worker = FunctionWorker(list_server_devices, self.cfg.server_url, self.cfg.server_dir)
        self._start_worker(worker, self._fill_server_devices, None, self.server_log, self._server_action_failed)

    def _fill_server_devices(self, devices: list[dict]) -> None:
        self.server_devices_table.setRowCount(0)
        for row, device in enumerate(devices):
            self.server_devices_table.insertRow(row)
            values = [
                device.get("device_name", ""),
                device.get("user_id", ""),
                "да" if device.get("is_online") else "нет",
                "включён" if device.get("monitoring_active") else "остановлен",
                device.get("client_version", ""),
                device.get("model_version", ""),
                device.get("last_seen_at", ""),
                "да" if device.get("is_blocked") else "нет",
            ]
            for col, value in enumerate(values):
                self.server_devices_table.setItem(row, col, QTableWidgetItem(str(value)))
        self.server_log.append(f"Получено устройств: {len(devices)}")
