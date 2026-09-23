"""Получение и экспорт агрегированной статистики."""

from __future__ import annotations

import csv
import io
import uuid
from datetime import datetime

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from ..dependencies import AdminActor, DbSession
from ..models import EmotionAggregate
from ..schemas import AggregateResponse


router = APIRouter(prefix="/admin/stats", tags=["statistics"])


@router.get("", response_model=list[AggregateResponse])
def get_statistics(
    db: DbSession,
    actor: AdminActor,
    user_id: uuid.UUID | None = None,
    period_type: str | None = Query(default=None, pattern="^(hour|day|week|month)$"),
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
    limit: int = Query(default=500, ge=1, le=5000),
) -> list[EmotionAggregate]:
    statement = select(EmotionAggregate)
    if user_id is not None:
        statement = statement.where(EmotionAggregate.user_id == user_id)
    if period_type is not None:
        statement = statement.where(EmotionAggregate.period_type == period_type)
    if start_utc is not None:
        statement = statement.where(EmotionAggregate.period_start_utc >= start_utc)
    if end_utc is not None:
        statement = statement.where(EmotionAggregate.period_start_utc < end_utc)
    statement = statement.order_by(EmotionAggregate.period_start_utc.desc()).limit(limit)
    return list(db.scalars(statement))


@router.get("/export.csv")
def export_statistics_csv(
    db: DbSession,
    actor: AdminActor,
    user_id: uuid.UUID | None = None,
    period_type: str | None = Query(default=None, pattern="^(hour|day|week|month)$"),
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
) -> StreamingResponse:
    statement = select(EmotionAggregate)
    if user_id is not None:
        statement = statement.where(EmotionAggregate.user_id == user_id)
    if period_type is not None:
        statement = statement.where(EmotionAggregate.period_type == period_type)
    if start_utc is not None:
        statement = statement.where(EmotionAggregate.period_start_utc >= start_utc)
    if end_utc is not None:
        statement = statement.where(EmotionAggregate.period_start_utc < end_utc)
    rows = list(db.scalars(statement.order_by(EmotionAggregate.period_start_utc)))
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "user_id",
        "period_type",
        "period_start_utc",
        "period_end_utc",
        "timezone_name",
        "dominant_emotion",
        "event_count",
        "speech_duration_sec",
        "average_confidence",
        "filtered_count",
        "volatility",
        "has_data",
        "probabilities",
        "emotion_shares",
    ])
    for row in rows:
        writer.writerow([
            row.user_id,
            row.period_type,
            row.period_start_utc.isoformat(),
            row.period_end_utc.isoformat(),
            row.timezone_name,
            row.dominant_emotion or "",
            row.event_count,
            row.speech_duration_sec,
            row.average_confidence,
            row.filtered_count,
            row.volatility,
            row.has_data,
            row.probabilities,
            row.emotion_shares,
        ])
    content = output.getvalue().encode("utf-8-sig")
    return StreamingResponse(
        iter([content]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=voiceemotion_statistics.csv"},
    )
