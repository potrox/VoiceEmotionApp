"""Создание FastAPI-приложения и подключение маршрутов."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from .config import get_settings
from .routers import admin, client, events, health, models_router, stats


settings = get_settings()
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
app = FastAPI(title=settings.app_name, version=settings.app_version)
app.include_router(health.router)
app.include_router(client.router, prefix=settings.api_prefix)
app.include_router(events.router, prefix=settings.api_prefix)
app.include_router(models_router.router, prefix=settings.api_prefix)
app.include_router(admin.router, prefix=settings.api_prefix)
app.include_router(stats.router, prefix=settings.api_prefix)
