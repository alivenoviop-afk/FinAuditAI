"""API: прогон файлов через модель, история прогонов."""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.config import settings
from app.core.pipeline import run_file
from app.db.models import RunRecord, SessionLocal
from app.providers.base import ProviderError

router = APIRouter(prefix="/api/runs", tags=["runs"])

ALLOWED_EXT = {".csv", ".jsonl", ".json", ".txt", ".md", ".xlsx", ".pdf", ".docx"}


class RunBody(BaseModel):
    provider: str
    model: str


@router.post("")
async def create_run(file: UploadFile = File(...), provider: str = "openai", model: str = ""):
    """Загрузка файла + запуск прогона в фоне. Возвращает run_id."""
    if not model:
        raise HTTPException(400, "Не указана модель")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Формат {ext} не поддерживается. Допустимо: {', '.join(sorted(ALLOWED_EXT))}")

    dest = settings.uploads_dir / f"{datetime.now(timezone.utc):%Y%m%d%H%M%S}_{Path(file.filename).name}"
    with dest.open("wb") as f:
        shutil_copy = file.file.read()
        f.write(shutil_copy)

    db = SessionLocal()
    run = RunRecord(model_name=f"{provider}:{model}", file_name=dest.name, status="running")
    db.add(run)
    db.commit()
    run_id = run.id
    db.close()

    def _bg():
        db = SessionLocal()
        rec = db.get(RunRecord, run_id)
        try:
            result = run_file(provider, model, dest)
            rec.result = result
            rec.status = "done"
        except ProviderError as e:
            rec.status = "failed"
            rec.error = str(e)
        except Exception as e:  # noqa: BLE001
            rec.status = "failed"
            rec.error = f"{type(e).__name__}: {e}"
        finally:
            rec.finished_at = datetime.now(timezone.utc)
            db.commit()
            db.close()

    threading.Thread(target=_bg, daemon=True).start()
    return {"ok": True, "run_id": run_id}


@router.get("")
def list_runs():
    db = SessionLocal()
    try:
        runs = db.query(RunRecord).order_by(RunRecord.id.desc()).limit(100).all()
        return {"runs": [{
            "id": r.id, "model": r.model_name, "file": r.file_name,
            "status": r.status, "error": r.error,
            "findings_count": len((r.result or {}).get("findings", [])),
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "finished_at": r.finished_at.isoformat() if r.finished_at else None,
        } for r in runs]}
    finally:
        db.close()


@router.get("/{run_id}")
def run_detail(run_id: int):
    db = SessionLocal()
    try:
        r = db.get(RunRecord, run_id)
        if r is None:
            raise HTTPException(404, "Прогон не найден")
        return {
            "id": r.id, "model": r.model_name, "file": r.file_name,
            "status": r.status, "error": r.error, "result": r.result,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
    finally:
        db.close()
