"""API: каталог моделей, подключение своего API-провайдера, health."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.db.models import ModelRecord, SessionLocal
from app.providers.base import ProviderError
from app.providers.registry import all_providers, collect_models, register_custom_api

router = APIRouter(prefix="/api/models", tags=["models"])


class CustomAPIBody(BaseModel):
    name: str
    base_url: str
    api_key: str = ""


@router.get("")
def list_models():
    """Единый список: локальные + API модели + ошибки недоступных провайдеров."""
    models, errors = collect_models()
    return {
        "models": [m.__dict__ for m in models],
        "errors": errors,
        "providers": [
            {"name": p.name, "kind": p.kind, "health": p.health()}
            for p in all_providers().values()
        ],
    }


@router.post("/connect")
def connect_api(body: CustomAPIBody):
    """Подключить свой OpenAI-совместимый API (для компании: внутренний endpoint)."""
    try:
        prov = register_custom_api(body.name, body.base_url, body.api_key)
        n = len(prov.list_models())
    except ProviderError as e:
        raise HTTPException(400, str(e)) from e

    db = SessionLocal()
    try:
        rec = db.query(ModelRecord).filter_by(name=body.name).first()
        if rec is None:
            rec = ModelRecord(name=body.name, provider=body.name, kind="api",
                              endpoint=body.base_url, can_train=False)
            db.add(rec)
        rec.is_active = True
        db.commit()
    finally:
        db.close()
    return {"ok": True, "provider": body.name, "models_found": n}
