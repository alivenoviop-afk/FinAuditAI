"""API: проверка модели через FinRepGYM (проверщик) + история оценок."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.evaluator import evaluate_model, setup_checker
from app.db.models import EvalResult, SessionLocal
from app.providers.base import ProviderError

router = APIRouter(prefix="/api/eval", tags=["eval"])


class EvalBody(BaseModel):
    provider: str
    model: str
    split: str = "dev"
    max_episodes: int = 10


@router.post("/setup")
def setup():
    """Клонирует FinRepGYM и прогоняет его собственную проверку (check.py)."""
    try:
        return setup_checker()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Ошибка настройки проверщика: {e}") from e


@router.post("/run")
def run_eval(body: EvalBody):
    """Прогон модели через эпизоды FinRepGYM: P/R/F1, reward."""
    try:
        metrics = evaluate_model(body.provider, body.model, body.split, body.max_episodes)
    except ProviderError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Ошибка прогона: {e}") from e

    db = SessionLocal()
    try:
        db.add(EvalResult(
            model_name=f"{body.provider}:{body.model}",
            episodes=body.split,
            precision=metrics["precision"], recall=metrics["recall"],
            f1=metrics["f1"], reward=metrics["reward"], steps=metrics["steps"],
            detail=metrics,
        ))
        db.commit()
    finally:
        db.close()
    return metrics


@router.get("/history")
def history():
    db = SessionLocal()
    try:
        rows = db.query(EvalResult).order_by(EvalResult.id.desc()).limit(50).all()
        return {"results": [{
            "id": r.id, "model": r.model_name, "split": r.episodes,
            "precision": r.precision, "recall": r.recall, "f1": r.f1,
            "reward": r.reward, "steps": r.steps,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        } for r in rows]}
    finally:
        db.close()
