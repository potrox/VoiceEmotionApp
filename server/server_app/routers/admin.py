"""Административные маршруты пользователей, устройств и моделей."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select

from ..dependencies import AdminActor, DbSession
from ..models import AggregationJob, Device, EmotionEvent, ModelVersion, User
from ..config import get_settings
from ..schemas import DeviceResponse, EventAdminResponse, EventDeleteRequest, ModelAdminResponse, RegistrationCodeCreate, RegistrationCodeResponse, UserResponse
from ..services.audit import add_audit
from ..services.models_service import activate_model, save_model_file
from ..services.registration import create_registration_code


router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/registration-codes", response_model=RegistrationCodeResponse)
def new_registration_code(request: RegistrationCodeCreate, db: DbSession, actor: AdminActor) -> RegistrationCodeResponse:
    code, expires_at = create_registration_code(db, request.expires_minutes, actor)
    return RegistrationCodeResponse(code=code, expires_at=expires_at)


@router.get("/users", response_model=list[UserResponse])
def list_users(db: DbSession, actor: AdminActor) -> list[User]:
    return list(db.scalars(select(User).where(User.deleted_at.is_(None)).order_by(User.created_at.desc())))


@router.get("/devices", response_model=list[DeviceResponse])
def list_devices(db: DbSession, actor: AdminActor) -> list[DeviceResponse]:
    devices = list(db.scalars(select(Device).where(Device.deleted_at.is_(None)).order_by(Device.created_at.desc())))
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=get_settings().device_online_seconds)
    return [
        DeviceResponse(
            id=device.id,
            user_id=device.user_id,
            device_name=device.device_name,
            client_version=device.client_version,
            model_version=device.model_version,
            model_sha256=device.model_sha256,
            last_seen_at=device.last_seen_at,
            monitoring_active=device.monitoring_active,
            is_blocked=device.is_blocked,
            is_online=bool(device.last_seen_at and device.last_seen_at >= cutoff and not device.is_blocked),
            created_at=device.created_at,
        )
        for device in devices
    ]


@router.get("/events", response_model=list[EventAdminResponse])
def list_events(
    db: DbSession,
    actor: AdminActor,
    user_id: uuid.UUID | None = None,
    include_deleted: bool = False,
    limit: int = 200,
) -> list[EmotionEvent]:
    limit = min(max(limit, 1), 5000)
    statement = select(EmotionEvent)
    if user_id is not None:
        statement = statement.where(EmotionEvent.user_id == user_id)
    if not include_deleted:
        statement = statement.where(EmotionEvent.deleted_at.is_(None))
    return list(db.scalars(statement.order_by(EmotionEvent.occurred_at.desc()).limit(limit)))


@router.get("/models", response_model=list[ModelAdminResponse])
def list_models(db: DbSession, actor: AdminActor) -> list[ModelVersion]:
    return list(db.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc())))


@router.post("/devices/{device_id}/block")
def block_device(device_id: uuid.UUID, db: DbSession, actor: AdminActor) -> dict:
    device = db.get(Device, device_id)
    if device is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Устройство не найдено")
    before = {"is_blocked": device.is_blocked}
    device.is_blocked = True
    add_audit(db, actor, "block_device", "device", str(device.id), before=before, after={"is_blocked": True})
    db.commit()
    return {"status": "ok"}


@router.post("/devices/{device_id}/unblock")
def unblock_device(device_id: uuid.UUID, db: DbSession, actor: AdminActor) -> dict:
    device = db.get(Device, device_id)
    if device is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Устройство не найдено")
    before = {"is_blocked": device.is_blocked}
    device.is_blocked = False
    add_audit(db, actor, "unblock_device", "device", str(device.id), before=before, after={"is_blocked": False})
    db.commit()
    return {"status": "ok"}


@router.delete("/events/{event_id}")
def soft_delete_event(event_id: uuid.UUID, request: EventDeleteRequest, db: DbSession, actor: AdminActor) -> dict:
    event = db.get(EmotionEvent, event_id)
    if event is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Событие не найдено")
    if event.deleted_at is None:
        event.deleted_at = datetime.now(timezone.utc)
        event.deletion_reason = request.reason
        add_audit(db, actor, "soft_delete_event", "emotion_event", str(event.id), after={"reason": request.reason})
        existing = db.scalar(select(AggregationJob).where(AggregationJob.event_id == event.id))
        if existing is None:
            db.add(AggregationJob(user_id=event.user_id, event_id=event.id))
        else:
            existing.status = "pending"
            existing.available_at = datetime.now(timezone.utc)
            existing.error_text = None
    db.commit()
    return {"status": "ok"}


@router.post("/events/{event_id}/restore")
def restore_event(event_id: uuid.UUID, db: DbSession, actor: AdminActor) -> dict:
    event = db.get(EmotionEvent, event_id)
    if event is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Событие не найдено")
    event.deleted_at = None
    event.deletion_reason = None
    add_audit(db, actor, "restore_event", "emotion_event", str(event.id))
    existing = db.scalar(select(AggregationJob).where(AggregationJob.event_id == event.id))
    if existing is None:
        db.add(AggregationJob(user_id=event.user_id, event_id=event.id))
    else:
        existing.status = "pending"
        existing.available_at = datetime.now(timezone.utc)
        existing.error_text = None
    db.commit()
    return {"status": "ok"}


@router.post("/models", status_code=status.HTTP_201_CREATED)
def upload_model(
    db: DbSession,
    actor: AdminActor,
    file: UploadFile = File(...),
    version: str = Form(...),
    display_name: str = Form(...),
    model_type: str = Form(...),
    config_json: str = Form("{}"),
    thresholds_json: str = Form("{}"),
) -> dict:
    if db.scalar(select(ModelVersion).where(ModelVersion.version == version)) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Версия модели уже существует")
    try:
        config = json.loads(config_json)
        thresholds = json.loads(thresholds_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Некорректный JSON параметров модели") from exc
    file_path, sha256, signature = save_model_file(file.file, version, file.filename or "model.bin")
    model = ModelVersion(
        version=version,
        display_name=display_name,
        model_type=model_type,
        file_name=Path(file.filename or "model.bin").name,
        file_path=file_path,
        sha256=sha256,
        signature=signature,
        config_json=config,
        thresholds_json=thresholds,
    )
    db.add(model)
    add_audit(db, actor, "upload_model", "model_version", version, after={"sha256": sha256})
    db.commit()
    db.refresh(model)
    return {"id": model.id, "version": model.version, "sha256": model.sha256, "signature": model.signature}


@router.post("/models/{version}/activate")
def publish_model(version: str, db: DbSession, actor: AdminActor) -> dict:
    model = db.scalar(select(ModelVersion).where(ModelVersion.version == version))
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Версия модели не найдена")
    activate_model(db, model, actor)
    return {"status": "ok", "active_version": version}
