"""Сохранение рассчитанных агрегатов эмоций в базе данных."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import EmotionAggregate, EmotionEvent
from .aggregation_core import PERIOD_TYPES, calculate_aggregate, period_bounds


def recompute_period(
    database_session: Session,
    user_id: uuid.UUID,
    timezone_name: str,
    period_type: str,
    period_start_utc: datetime,
    period_end_utc: datetime,
) -> EmotionAggregate:
    events = list(
        database_session.scalars(
            select(EmotionEvent)
            .where(
                EmotionEvent.user_id == user_id,
                EmotionEvent.occurred_at >= period_start_utc,
                EmotionEvent.occurred_at < period_end_utc,
                EmotionEvent.deleted_at.is_(None),
            )
            .order_by(EmotionEvent.occurred_at)
        )
    )
    aggregate_values = calculate_aggregate(
        events,
        period_type,
        get_settings().minimum_hour_speech_seconds,
    )
    aggregate = database_session.scalar(
        select(EmotionAggregate).where(
            EmotionAggregate.user_id == user_id,
            EmotionAggregate.period_type == period_type,
            EmotionAggregate.period_start_utc == period_start_utc,
            EmotionAggregate.timezone_name == timezone_name,
        )
    )
    if aggregate is None:
        aggregate = EmotionAggregate(
            user_id=user_id,
            period_type=period_type,
            period_start_utc=period_start_utc,
            period_end_utc=period_end_utc,
            timezone_name=timezone_name,
        )
        database_session.add(aggregate)
    for field_name, field_value in aggregate_values.items():
        setattr(aggregate, field_name, field_value)
    aggregate.period_end_utc = period_end_utc
    aggregate.updated_at = datetime.now(timezone.utc)
    return aggregate


def recompute_for_event(database_session: Session, event: EmotionEvent) -> None:
    for period_type in PERIOD_TYPES:
        period_start, period_end = period_bounds(
            event.occurred_at, event.timezone_name, period_type
        )
        recompute_period(
            database_session,
            event.user_id,
            event.timezone_name,
            period_type,
            period_start,
            period_end,
        )
