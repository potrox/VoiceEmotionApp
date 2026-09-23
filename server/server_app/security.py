"""Хеширование, шифрование и проверка серверных ключей."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

from cryptography.fernet import Fernet, InvalidToken

from .config import get_settings


def generate_registration_code() -> str:
    return secrets.token_urlsafe(18)


def generate_device_token() -> str:
    return secrets.token_urlsafe(32)


def secret_hash(value: str) -> str:
    settings = get_settings()
    return hmac.new(settings.token_secret.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()


def sign_model(sha256: str, version: str) -> str:
    settings = get_settings()
    payload = f"{version}:{sha256}".encode("utf-8")
    return hmac.new(settings.model_signing_key.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def verify_admin_key(candidate: str) -> bool:
    settings = get_settings()
    return hmac.compare_digest(settings.admin_api_key, candidate)


def _fernet() -> Fernet:
    secret_bytes = get_settings().data_encryption_key.encode("utf-8")
    key = base64.urlsafe_b64encode(hashlib.sha256(secret_bytes).digest())
    return Fernet(key)


def encrypt_text(value: str) -> str:
    return _fernet().encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_text(value: str) -> str:
    try:
        return _fernet().decrypt(value.encode("ascii")).decode("utf-8")
    except InvalidToken:
        return "[данные недоступны]"


def blind_index(value: str) -> str:
    normalized = value.strip().casefold().encode("utf-8")
    return hmac.new(get_settings().data_encryption_key.encode("utf-8"), normalized, hashlib.sha256).hexdigest()
