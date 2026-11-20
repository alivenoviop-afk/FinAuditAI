"""Фоновое обучение моделей (LoRA) с прогрессом и логами в БД."""
from __future__ import annotations

import threading
from datetime import datetime, timezone

from app.db.models import SessionLocal, TrainingJob
from app.providers.registry import get_provider


def start_training(job_id: int) -> None:
    """Запуск задачи обучения в отдельном потоке (для масштаба — Celery/Redis)."""
    threading.Thread(target=_run, args=(job_id,), daemon=True).start()


def _run(job_id: int) -> None:
    db = SessionLocal()
    job = db.get(TrainingJob, job_id)
    if job is None:
        db.close()
        return
    job.status = "running"
    db.commit()

    def on_log(msg: str) -> None:
        job.logs = (job.logs + msg + "\n")[-20000:]
        db.commit()

    def on_progress(p: float) -> None:
        job.progress = p
        db.commit()

    try:
        model_name = job.result.get("provider_model") or _resolve_model(job)
        prov_name, _, provider_model = model_name.partition("/")
        # провайдер определяем по полю модели: "provider:model"
        provider_key, sep, base_model = model_name.partition(":")
        if not sep:
            provider_key, base_model = "huggingface", model_name
        prov = get_provider(provider_key)
        on_log(f"Провайдер: {provider_key}, модель: {base_model}")
        result = prov.train(base_model, job.dataset_path, on_log=on_log, on_progress=on_progress)
        job.result = {**(job.result or {}), **result, "model": base_model}
        job.status = "done"
        job.progress = 1.0
    except Exception as e:  # noqa: BLE001 — ошибки уходят в UI
        job.status = "failed"
        on_log(f"ОШИБКА: {e}")
    finally:
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
        db.close()


def _resolve_model(job: TrainingJob) -> str:
    if job.model_id and job.result.get("model"):
        return job.result["model"]
    return job.result.get("model", "")
