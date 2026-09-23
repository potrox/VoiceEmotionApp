"""Зависимости FastAPI для базы данных и авторизации."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import get_db
from .models import Device
from .security import secret_hash, verify_admin_key


bearer_scheme = HTTPBearer(auto_error=False)
DbSession = Annotated[Session, Depends(get_db)]


def require_admin(x_admin_key: Annotated[str | None, Header()] = None) -> str:
    if not x_admin_key or not verify_admin_key(x_admin_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Неверный ключ администратора")
    return "admin"


def require_device(
    db: DbSession,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> Device:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Необходим токен устройства")
    token_hash = secret_hash(credentials.credentials)
    device = db.scalar(select(Device).where(Device.token_hash == token_hash, Device.deleted_at.is_(None)))
    if device is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Неизвестное устройство")
    if device.is_blocked:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Устройство заблокировано")
    return device


AdminActor = Annotated[str, Depends(require_admin)]
CurrentDevice = Annotated[Device, Depends(require_device)]
