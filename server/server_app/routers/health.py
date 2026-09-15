"""Проверка доступности API и базы данных."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import text

from ..config import get_settings
from ..dependencies import DbSession


router = APIRouter(tags=["health"])


@router.get("/health")
def health(db: DbSession) -> dict:
    db.execute(text("SELECT 1"))
    settings = get_settings()
    return {"status": "ok", "service": settings.app_name, "version": settings.app_version}
