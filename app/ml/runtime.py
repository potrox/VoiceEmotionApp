"""Настройки вычислительных ресурсов для моделей sklearn."""

import os

def sklearn_n_jobs() -> int:
    try:
        value = int(os.environ.get("VOICE_EMOTION_SKLEARN_N_JOBS", "1"))
    except ValueError:
        value = 1
    return value if value != 0 else 1
