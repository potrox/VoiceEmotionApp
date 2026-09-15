"""Регистрация клиентов и обновление состояния устройств."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, status

from ..dependencies import CurrentDevice, DbSession
from ..models import User
from ..security import blind_index, encrypt_text
from ..schemas import (
    ClientRegisterRequest,
    ClientRegisterResponse,
    HeartbeatRequest,
    UserSwitchRequest,
)
from ..services.registration import register_client


router = APIRouter(prefix="/client", tags=["client"])


@router.post("/register", response_model=ClientRegisterResponse, status_code=status.HTTP_201_CREATED)
def register(request: ClientRegisterRequest, db: DbSession) -> ClientRegisterResponse:
    user, device, token = register_client(db, request)
    return ClientRegisterResponse(
        user_id=user.id,
        device_id=device.id,
        token=token,
        display_name=user.display_name,
    )


@router.post("/heartbeat")
def heartbeat(request: HeartbeatRequest, db: DbSession, device: CurrentDevice) -> dict:
    device.last_seen_at = datetime.now(timezone.utc)
    device.client_version = request.client_version
    device.model_version = request.model_version
    device.model_sha256 = request.model_sha256
    device.monitoring_active = request.monitoring_active
    db.commit()
    return {"status": "ok", "server_time": datetime.now(timezone.utc)}


@router.post("/switch-user")
def switch_user(request: UserSwitchRequest, db: DbSession, device: CurrentDevice) -> dict:
    display_name = request.display_name.strip()
    user = User(
        display_name_encrypted=encrypt_text(display_name),
        display_name_hash=blind_index(display_name),
        timezone_name=request.timezone_name,
    )
    db.add(user)
    db.flush()
    device.user_id = user.id
    device.last_seen_at = datetime.now(timezone.utc)
    db.commit()
    return {"user_id": user.id, "display_name": user.display_name}
