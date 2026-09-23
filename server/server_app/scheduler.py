"""Периодический пересчёт агрегатов и очистка устаревших данных."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete

from .config import get_settings
from .database import get_session_factory
from .models import AggregationJob, AuditLog, EmotionAggregate, EmotionEvent, RegistrationCode


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voiceemotion.scheduler")


def cleanup() -> None:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    event_cutoff = now - timedelta(days=settings.event_retention_days)
    aggregate_cutoff = now - timedelta(days=settings.aggregate_retention_days)
    audit_cutoff = now - timedelta(days=settings.audit_retention_days)
    job_cutoff = now - timedelta(days=7)
    code_cutoff = now - timedelta(days=30)
    factory = get_session_factory()
    with factory() as db:
        db.execute(delete(EmotionEvent).where(EmotionEvent.occurred_at < event_cutoff))
        db.execute(delete(EmotionAggregate).where(EmotionAggregate.period_end_utc < aggregate_cutoff))
        db.execute(delete(AuditLog).where(AuditLog.created_at < audit_cutoff))
        db.execute(delete(AggregationJob).where(AggregationJob.status == "done", AggregationJob.updated_at < job_cutoff))
        db.execute(delete(RegistrationCode).where(RegistrationCode.expires_at < code_cutoff))
        db.commit()


def main() -> None:
    settings = get_settings()
    while True:
        try:
            cleanup()
        except Exception:
            logger.exception("Ошибка плановой очистки")
        time.sleep(settings.scheduler_poll_seconds)


if __name__ == "__main__":
    main()
