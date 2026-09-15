"""Проверки примера конфигурации приложения."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from app.config import AppConfig


def test_example_config_matches_application_defaults() -> None:
    project_root = Path(__file__).resolve().parents[1]
    example_settings = json.loads(
        (project_root / "config.example.json").read_text(encoding="utf-8")
    )

    assert example_settings == asdict(AppConfig())
    assert example_settings["input_device_index"] is None
    assert example_settings["server_dir"] == "server"
