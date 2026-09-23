"""Хранение, подпись и активация версий моделей."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

from fastapi import HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import ModelDeployment, ModelVersion
from ..security import sign_model
from .audit import add_audit


MODEL_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def validate_model_version(version: str) -> str:
    if not MODEL_VERSION_PATTERN.fullmatch(version):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Версия модели может содержать только буквы ASCII, цифры, точку, дефис и подчёркивание",
        )
    return version


def save_model_file(source: BinaryIO, version: str, original_name: str) -> tuple[str, str, str]:
    settings = get_settings()
    version = validate_model_version(version)
    target_dir = Path(settings.models_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(original_name).name
    target = target_dir / f"{version}_{safe_name}"
    digest = hashlib.sha256()
    with target.open("wb") as output:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            output.write(chunk)
    sha256 = digest.hexdigest()
    signature = sign_model(sha256, version)
    return str(target), sha256, signature


def activate_model(db: Session, model: ModelVersion, actor: str, notes: str | None = None) -> ModelDeployment:
    now = datetime.now(timezone.utc)
    active = list(db.scalars(select(ModelDeployment).where(ModelDeployment.is_active.is_(True))))
    for deployment in active:
        deployment.is_active = False
        deployment.deactivated_at = now
    db.execute(update(ModelVersion).where(ModelVersion.is_published.is_(True)).values(is_published=False, deactivated_at=now))
    model.is_published = True
    model.published_at = now
    model.deactivated_at = None
    deployment = ModelDeployment(model_version_id=model.id, is_active=True, notes=notes)
    db.add(deployment)
    add_audit(db, actor, "activate_model", "model_version", str(model.id), after={"version": model.version})
    db.commit()
    db.refresh(deployment)
    return deployment


def get_active_model(db: Session) -> ModelVersion | None:
    deployment = db.scalar(
        select(ModelDeployment)
        .where(ModelDeployment.is_active.is_(True))
        .order_by(ModelDeployment.activated_at.desc())
    )
    if deployment is None:
        return None
    return db.get(ModelVersion, deployment.model_version_id)


def require_model(db: Session, version: str) -> ModelVersion:
    model = db.scalar(select(ModelVersion).where(ModelVersion.version == version))
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Версия модели не найдена")
    return model
