"""Приём результатов распознавания от клиентских устройств."""

from __future__ import annotations

from fastapi import APIRouter

from ..dependencies import CurrentDevice, DbSession
from sqlalchemy import select

from ..models import ModelVersion
from ..schemas import EmotionEventBatchRequest, EmotionEventBatchResponse
from ..services.events import insert_event, touch_device
from ..services.models_service import get_active_model


router = APIRouter(prefix="/client/events", tags=["events"])


@router.post("/batch", response_model=EmotionEventBatchResponse)
def receive_events(request: EmotionEventBatchRequest, db: DbSession, device: CurrentDevice) -> EmotionEventBatchResponse:
    accepted_ids = []
    duplicate_ids = []
    rejected_ids = []
    last_item = None
    active_model = get_active_model(db)
    model_cache: dict[str, ModelVersion | None] = {}
    for item in request.events:
        last_item = item
        if item.model_version not in model_cache:
            model_cache[item.model_version] = db.scalar(
                select(ModelVersion).where(
                    ModelVersion.version == item.model_version,
                    ModelVersion.sha256 == item.model_sha256,
                )
            )
        event, duplicate, rejected_by_policy = insert_event(
            db,
            device,
            item,
            active_model,
            model_cache[item.model_version],
        )
        if duplicate:
            duplicate_ids.append(item.message_id)
        elif rejected_by_policy or event is None:
            rejected_ids.append(item.message_id)
        else:
            accepted_ids.append(item.message_id)
    if last_item is not None:
        touch_device(db, device, last_item.client_version, last_item.model_version, last_item.model_sha256)
    db.commit()
    return EmotionEventBatchResponse(
        accepted=len(accepted_ids),
        duplicates=len(duplicate_ids),
        rejected=len(rejected_ids),
        accepted_message_ids=accepted_ids,
        duplicate_message_ids=duplicate_ids,
        rejected_message_ids=rejected_ids,
    )
