"""Управление локальным Docker-сервером и его административным API."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import joblib
import requests


class ServerIntegrationError(RuntimeError):
    pass


def _creation_flags() -> int:
    if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        return subprocess.CREATE_NO_WINDOW
    return 0


def read_env_file(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.is_file():
        raise ServerIntegrationError(f"Файл .env не найден: {env_path}")
    result: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        result[name.strip()] = value.strip()
    return result


def validate_server_directory(server_dir: str | Path) -> Path:
    path = Path(server_dir).expanduser().resolve()
    if not path.is_dir():
        raise ServerIntegrationError(f"Папка сервера не найдена: {path}")
    compose = path / "compose.yaml"
    if not compose.is_file():
        raise ServerIntegrationError(f"В папке сервера отсутствует compose.yaml: {compose}")
    return path


def admin_key_from_server_dir(server_dir: str | Path) -> str:
    path = validate_server_directory(server_dir)
    values = read_env_file(path / ".env")
    key = values.get("ADMIN_API_KEY", "").strip()
    if len(key) < 32:
        raise ServerIntegrationError("ADMIN_API_KEY отсутствует или слишком короткий в серверном .env")
    return key


def run_command(command: list[str], cwd: str | Path, timeout: float = 900.0) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            creationflags=_creation_flags(),
            check=False,
        )
    except FileNotFoundError as exc:
        raise ServerIntegrationError(f"Команда не найдена: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ServerIntegrationError(f"Команда выполнялась дольше {timeout:.0f} секунд") from exc
    output = "\n".join(part.strip() for part in [completed.stdout, completed.stderr] if part.strip())
    if completed.returncode != 0:
        raise ServerIntegrationError(output or f"Команда завершилась с кодом {completed.returncode}")
    return output or "Команда выполнена успешно."


def docker_compose(server_dir: str | Path, args: list[str], timeout: float = 900.0) -> str:
    path = validate_server_directory(server_dir)
    return run_command(["docker", "compose", *args], path, timeout=timeout)


def verify_server_environment(server_dir: str | Path) -> str:
    path = validate_server_directory(server_dir)
    values = read_env_file(path / ".env")
    required = [
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "APP_DB_USER",
        "APP_DB_PASSWORD",
        "DATABASE_URL",
        "ADMIN_API_KEY",
        "TOKEN_SECRET",
        "MODEL_SIGNING_KEY",
        "DATA_ENCRYPTION_KEY",
        "BACKUP_ENCRYPTION_KEY",
    ]
    missing = [name for name in required if not values.get(name, "").strip()]
    if missing:
        raise ServerIntegrationError("В .env отсутствуют значения: " + ", ".join(missing))
    return "Серверная папка и файл .env заполнены корректно."


def server_status(server_dir: str | Path, server_url: str) -> str:
    lines = [docker_compose(server_dir, ["ps", "-a"], timeout=60.0)]
    try:
        response = requests.get(server_url.rstrip("/") + "/health", timeout=5.0)
        response.raise_for_status()
        lines.append("API: " + json.dumps(response.json(), ensure_ascii=False))
    except Exception as exc:
        lines.append(f"API недоступен: {exc}")
    return "\n\n".join(lines)


def start_server(server_dir: str | Path, rebuild: bool = False) -> str:
    outputs: list[str] = []
    docker_compose(server_dir, ["config"], timeout=60.0)
    if rebuild:
        outputs.append(docker_compose(server_dir, ["build"], timeout=3600.0))
    outputs.append(docker_compose(server_dir, ["up", "-d"], timeout=900.0))
    outputs.append(docker_compose(server_dir, ["ps", "-a"], timeout=60.0))
    return "\n\n".join(outputs)


def stop_server(server_dir: str | Path) -> str:
    return docker_compose(server_dir, ["stop"], timeout=300.0)


def restart_server(server_dir: str | Path) -> str:
    outputs = [docker_compose(server_dir, ["restart"], timeout=600.0)]
    outputs.append(docker_compose(server_dir, ["ps", "-a"], timeout=60.0))
    return "\n\n".join(outputs)


def run_migrations(server_dir: str | Path) -> str:
    docker_compose(server_dir, ["rm", "-f", "migrate"], timeout=120.0)
    return docker_compose(server_dir, ["up", "--no-deps", "migrate"], timeout=900.0)


def service_logs(server_dir: str | Path, service: str, tail: int = 300) -> str:
    allowed = {"db", "api", "worker", "scheduler", "backup", "migrate"}
    if service not in allowed:
        raise ServerIntegrationError(f"Неизвестный сервис: {service}")
    return docker_compose(server_dir, ["logs", "--no-color", "--tail", str(max(1, tail)), service], timeout=120.0)


def create_backup(server_dir: str | Path) -> str:
    path = validate_server_directory(server_dir)
    values = read_env_file(path / ".env")
    user = values.get("POSTGRES_USER", "voiceemotion_admin")
    database = values.get("POSTGRES_DB", "voiceemotion")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"voiceemotion_ui_{stamp}.dump"
    output = docker_compose(
        path,
        ["exec", "-T", "db", "pg_dump", "-U", user, "-d", database, "-Fc", "-f", f"/backups/{file_name}"],
        timeout=900.0,
    )
    host_path = path / "backups" / file_name
    return f"{output}\nРезервная копия: {host_path}"


def _powershell_executable() -> str:
    candidates = [
        shutil.which("pwsh"),
        shutil.which("powershell"),
    ]
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    candidates.append(str(Path(system_root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"))
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise ServerIntegrationError("PowerShell не найден")


def run_server_test(server_dir: str | Path) -> str:
    path = validate_server_directory(server_dir)
    script = path / "test_server.ps1"
    if not script.is_file():
        raise ServerIntegrationError(f"Скрипт тестирования не найден: {script}")
    return run_command(
        [_powershell_executable(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        path,
        timeout=900.0,
    )


def admin_request(
    server_url: str,
    server_dir: str | Path,
    method: str,
    endpoint: str,
    timeout: float = 30.0,
    **kwargs: Any,
) -> requests.Response:
    server_url = validate_admin_url(server_url)
    key = admin_key_from_server_dir(server_dir)
    headers = dict(kwargs.pop("headers", {}))
    headers["X-Admin-Key"] = key
    url = server_url.rstrip("/") + endpoint
    try:
        response = requests.request(method, url, headers=headers, timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        raise ServerIntegrationError(str(exc)) from exc
    if response.status_code >= 400:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except Exception:
            pass
        raise ServerIntegrationError(f"HTTP {response.status_code}: {detail}")
    return response


def validate_admin_url(server_url: str) -> str:
    value = server_url.strip().rstrip("/")
    parsed = urlsplit(value)
    if not parsed.hostname or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
        raise ServerIntegrationError("Адрес API должен быть адресом сервера без учётных данных и пути")
    if parsed.scheme == "https":
        return value
    if parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        return value
    raise ServerIntegrationError("Административный ключ можно отправлять только через HTTPS или локальный HTTP")


def create_registration_code(server_url: str, server_dir: str | Path, expires_minutes: int) -> dict[str, Any]:
    response = admin_request(
        server_url,
        server_dir,
        "POST",
        "/api/v1/admin/registration-codes",
        json={"expires_minutes": int(expires_minutes)},
    )
    return response.json()


def list_server_models(server_url: str, server_dir: str | Path) -> list[dict[str, Any]]:
    response = admin_request(server_url, server_dir, "GET", "/api/v1/admin/models")
    return list(response.json())


def activate_server_model(server_url: str, server_dir: str | Path, version: str) -> dict[str, Any]:
    response = admin_request(server_url, server_dir, "POST", f"/api/v1/admin/models/{quote(version, safe='')}/activate")
    return response.json()


def list_server_devices(server_url: str, server_dir: str | Path) -> list[dict[str, Any]]:
    response = admin_request(server_url, server_dir, "GET", "/api/v1/admin/devices")
    return list(response.json())


def _normalize_model_name(name: str) -> str:
    return str(name or "unknown").replace("RandomForest", "Random Forest")


def model_publication_metadata(model_path: str | Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
    path = Path(model_path)
    if not path.is_file():
        raise ServerIntegrationError(f"Файл модели не найден: {path}")
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise ServerIntegrationError(f"Не удалось прочитать пакет модели: {exc}") from exc
    if not isinstance(bundle, dict):
        raise ServerIntegrationError("Неподдерживаемый формат модели: ожидался словарь TrainingBundle")
    best_name = _normalize_model_name(bundle.get("best_model_name", "unknown"))
    preprocessing = dict(bundle.get("preprocessing_params") or {})
    feature_names = list(bundle.get("feature_names") or [])
    metrics = dict(bundle.get("metrics") or {})
    dataset_info = dict(bundle.get("dataset_info") or {})
    label_encoder = bundle.get("label_encoder")
    emotions: list[str] = []
    try:
        emotions = [str(value) for value in label_encoder.classes_]
    except Exception:
        emotions = ["joy", "sadness", "anger", "calm"]
    config = {
        "format_version": 1,
        "sample_rate": int(preprocessing.get("sample_rate", 16000)),
        "preprocessing": preprocessing,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "emotions": emotions,
        "best_model_name": best_name,
        "created_at": str(bundle.get("created_at", "")),
        "dataset_info": dataset_info,
        "metrics": metrics,
    }
    thresholds = {
        "default": 0.65,
        "margin": 0.10,
        "short_segment_bonus": 0.10,
        "speaker_playback_bonus": 0.15,
        "per_emotion": {},
        "calibration": "fixed_heuristic",
    }
    return best_name, config, thresholds


def publish_model(
    server_url: str,
    server_dir: str | Path,
    model_path: str | Path,
    version: str,
    display_name: str,
    activate: bool,
) -> dict[str, Any]:
    version = version.strip()
    display_name = display_name.strip()
    if not version:
        raise ServerIntegrationError("Укажите версию модели")
    if not display_name:
        raise ServerIntegrationError("Укажите отображаемое название модели")
    model_type, config, thresholds = model_publication_metadata(model_path)
    path = Path(model_path)
    with path.open("rb") as stream:
        response = admin_request(
            server_url,
            server_dir,
            "POST",
            "/api/v1/admin/models",
            timeout=300.0,
            files={"file": (path.name, stream, "application/octet-stream")},
            data={
                "version": version,
                "display_name": display_name,
                "model_type": model_type,
                "config_json": json.dumps(config, ensure_ascii=False, default=str),
                "thresholds_json": json.dumps(thresholds, ensure_ascii=False, default=str),
            },
        )
    result = response.json()
    if activate:
        activation = activate_server_model(server_url, server_dir, version)
        result["activation"] = activation
    return result


def open_url(url: str) -> None:
    webbrowser.open(url)


def open_directory(path: str | Path) -> None:
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(str(directory))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(directory)])
    else:
        subprocess.Popen(["xdg-open", str(directory)])
