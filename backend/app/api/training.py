"""API: загрузка датасетов, задачи обучения (LoRA), логи и прогресс."""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.config import settings
from app.core.trainer import start_training
from app.db.models import SessionLocal, TrainingJob
from app.providers.base import ProviderError
from app.providers.registry import get_provider

router = APIRouter(prefix="/api/training", tags=["training"])

ALLOWED_DATASET_EXT = {".jsonl", ".json", ".csv", ".txt"}


class StartBody(BaseModel):
    provider: str = "huggingface"
    model: str
    dataset: str            # имя файла в data/datasets
    epochs_hint: str = ""


@router.get("/datasets")
def list_datasets():
    files = []
    if settings.datasets_dir.exists():
        for p in sorted(settings.datasets_dir.glob("*")):
            if p.is_file():
                lines = sum(1 for _ in p.open(encoding="utf-8", errors="replace")) if p.suffix in (".jsonl", ".txt") else None
                files.append({"name": p.name, "size": p.stat().st_size, "lines": lines})
    return {"datasets": files}


@router.post("/datasets")
async def upload_dataset(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_DATASET_EXT:
        raise HTTPException(400, f"Формат {ext} не подходит. Нужен: {', '.join(sorted(ALLOWED_DATASET_EXT))}")
    dest = settings.datasets_dir / Path(file.filename).name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"ok": True, "name": dest.name, "size": dest.stat().st_size}


@router.post("/jobs")
def start_job(body: StartBody):
    """Запуск fine-tuning (LoRA) в фоне."""
    dataset_path = settings.datasets_dir / body.dataset
    if not dataset_path.exists():
        raise HTTPException(404, f"Датасет {body.dataset} не найден")
    try:
        prov = get_provider(body.provider)
        if not prov.can_train(body.model):
            raise HTTPException(400, f"Провайдер {body.provider} не обучает модели — выбери локальную (HuggingFace)")
    except ProviderError as e:
        raise HTTPException(400, str(e)) from e

    db = SessionLocal()
    try:
        job = TrainingJob(
            dataset_path=str(dataset_path),
            result={"provider": body.provider, "model": body.model,
                    "provider_model": f"{body.provider}:{body.model}"},
        )
        db.add(job)
        db.commit()
        job_id = job.id
    finally:
        db.close()

    start_training(job_id)
    return {"ok": True, "job_id": job_id}


@router.get("/jobs")
def list_jobs():
    db = SessionLocal()
    try:
        jobs = db.query(TrainingJob).order_by(TrainingJob.id.desc()).limit(50).all()
        return {"jobs": [{
            "id": j.id, "status": j.status, "progress": j.progress,
            "dataset": Path(j.dataset_path).name,
            "model": (j.result or {}).get("model", ""),
            "provider": (j.result or {}).get("provider", ""),
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "finished_at": j.finished_at.isoformat() if j.finished_at else None,
            "result": j.result,
        } for j in jobs]}
    finally:
        db.close()


@router.get("/jobs/{job_id}")
def job_detail(job_id: int):
    db = SessionLocal()
    try:
        j = db.get(TrainingJob, job_id)
        if j is None:
            raise HTTPException(404, "Задача не найдена")
        return {
            "id": j.id, "status": j.status, "progress": j.progress,
            "logs": j.logs, "result": j.result,
            "dataset": Path(j.dataset_path).name,
        }
    finally:
        db.close()
