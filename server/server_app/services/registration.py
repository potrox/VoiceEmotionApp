"""Регистрация устройств по одноразовым кодам."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Device, RegistrationCode, User
from ..schemas import ClientRegisterRequest
from ..security import blind_index, encrypt_text, generate_device_token, generate_registration_code, secret_hash


def create_registration_code(db: Session, expires_minutes: int, created_by: str) -> tuple[str, datetime]:
    raw_code = generate_registration_code()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    db.add(
        RegistrationCode(
            code_hash=secret_hash(raw_code),
            expires_at=expires_at,
            created_by=created_by,
        )
    )
    db.commit()
    return raw_code, expires_at


def register_client(db: Session, request: ClientRegisterRequest) -> tuple[User, Device, str]:
    now = datetime.now(timezone.utc)
    code = db.scalar(
        select(RegistrationCode).where(
            RegistrationCode.code_hash == secret_hash(request.code),
            RegistrationCode.used_at.is_(None),
            RegistrationCode.expires_at > now,
        )
    )
    if code is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Код регистрации неверен, использован или просрочен")
    display_name = request.display_name.strip()
    user = User(
        display_name_encrypted=encrypt_text(display_name),
        display_name_hash=blind_index(display_name),
        timezone_name=request.timezone_name,
    )
    token = generate_device_token()
    device = Device(
        user=user,
        device_name_encrypted=encrypt_text(request.device_name.strip()),
        token_hash=secret_hash(token),
        client_version=request.client_version,
        last_seen_at=now,
    )
    db.add_all([user, device])
    db.flush()
    code.used_at = now
    code.used_by_device_id = device.id
    db.commit()
    db.refresh(user)
    db.refresh(device)
    return user, device, token
