"""Фоновый обработчик очереди агрегирования."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from .config import get_settings
from .database import get_session_factory
from .models import AggregationJob, EmotionEvent
from .services.aggregation import recompute_for_event


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voiceemotion.worker")


def process_jobs() -> int:
    factory = get_session_factory()
    processed = 0
    with factory() as db:
        jobs = list(
            db.scalars(
                select(AggregationJob)
                .where(
                    AggregationJob.status == "pending",
                    AggregationJob.available_at <= datetime.now(timezone.utc),
                )
                .order_by(AggregationJob.id)
                .with_for_update(skip_locked=True)
                .limit(100)
            )
        )
        for job in jobs:
            job.status = "processing"
            job.attempts += 1
            try:
                event = db.get(EmotionEvent, job.event_id)
                if event is not None:
                    recompute_for_event(db, event)
                job.status = "done"
                job.error_text = None
                processed += 1
            except Exception as exc:
                logger.exception("Ошибка агрегации события %s", job.event_id)
                job.status = "pending" if job.attempts < 5 else "failed"
                job.available_at = datetime.now(timezone.utc) + timedelta(seconds=min(300, 2 ** job.attempts))
                job.error_text = str(exc)[:4000]
        db.commit()
    return processed


def main() -> None:
    settings = get_settings()
    while True:
        count = process_jobs()
        if count == 0:
            time.sleep(settings.worker_poll_seconds)


if __name__ == "__main__":
    main()
