"""Получение и загрузка активной модели клиентом."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from ..config import get_settings
from ..dependencies import CurrentDevice, DbSession
from ..schemas import ModelAckRequest, ModelMetadataResponse
from ..services.models_service import get_active_model, require_model


router = APIRouter(prefix="/client/models", tags=["models"])


@router.get("/current", response_model=ModelMetadataResponse)
def current_model(db: DbSession, device: CurrentDevice) -> ModelMetadataResponse:
    model = get_active_model(db)
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Активная модель не назначена")
    settings = get_settings()
    return ModelMetadataResponse(
        version=model.version,
        display_name=model.display_name,
        model_type=model.model_type,
        sha256=model.sha256,
        signature=model.signature,
        config=model.config_json,
        thresholds=model.thresholds_json,
        download_url=f"{settings.api_prefix}/client/models/{model.version}/download",
    )


@router.get("/{version}/download")
def download_model(version: str, db: DbSession, device: CurrentDevice) -> FileResponse:
    model = require_model(db, version)
    path = Path(model.file_path)
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Файл модели отсутствует")
    return FileResponse(path, media_type="application/octet-stream", filename=model.file_name)


@router.post("/{version}/ack")
def acknowledge_model(version: str, request: ModelAckRequest, db: DbSession, device: CurrentDevice) -> dict:
    model = require_model(db, version)
    if request.success and request.sha256 != model.sha256:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Контрольная сумма не совпадает")
    if request.success:
        device.model_version = version
        device.model_sha256 = request.sha256
    db.commit()
    return {"status": "ok", "success": request.success}
