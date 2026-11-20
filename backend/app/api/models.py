"""API: каталог моделей, подключение API-ключей и своего провайдера, health."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import ROOT_DIR, settings
from app.db.models import ModelRecord, SessionLocal
from app.providers.base import ProviderError
from app.providers.registry import all_providers, collect_models, register_custom_api, set_provider_keys

router = APIRouter(prefix="/api/models", tags=["models"])


class CustomAPIBody(BaseModel):
    name: str
    base_url: str
    api_key: str = ""


class KeysBody(BaseModel):
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    anthropic_api_key: str | None = None


def _persist_env(updates: dict[str, str]) -> None:
    """Записывает ключи в backend/.env — переживают перезапуск сервера."""
    env_path = Path(ROOT_DIR) / ".env"
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    for key, value in updates.items():
        replaced = False
        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}={value}"
                replaced = True
                break
        if not replaced:
            lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@router.get("")
def list_models():
    """Единый список: локальные + API модели + статусы провайдеров."""
    models, errors = collect_models()
    return {
        "models": [m.__dict__ for m in models],
        "errors": errors,
        "providers": [provider_status(p) for p in all_providers().values()],
    }


def provider_status(p) -> dict:
    """Статус провайдера: ok | no_key | offline (вместо красной ошибки)."""
    if p.kind == "api" and not getattr(p, "api_key", ""):
        return {"name": p.name, "kind": p.kind, "health": False, "status": "no_key"}
    try:
        ok = p.health()
    except Exception:  # noqa: BLE001
        ok = False
    if ok:
        status = "ok"
    elif p.kind == "local":
        status = "offline"
    else:
        status = "no_key" if not getattr(p, "api_key", "") else "error"
    return {"name": p.name, "kind": p.kind, "health": ok, "status": status}


@router.post("/keys")
def save_keys(body: KeysBody):
    """Сохранить API-ключи OpenAI/Anthropic: сразу действуют + пишутся в .env."""
    updates: dict[str, str] = {}
    if body.openai_api_key is not None:
        updates["OPENAI_API_KEY"] = body.openai_api_key.strip()
        settings.openai_api_key = updates["OPENAI_API_KEY"]
    if body.openai_base_url is not None and body.openai_base_url.strip():
        updates["OPENAI_BASE_URL"] = body.openai_base_url.strip()
        settings.openai_base_url = updates["OPENAI_BASE_URL"]
    if body.anthropic_api_key is not None:
        updates["ANTHROPIC_API_KEY"] = body.anthropic_api_key.strip()
        settings.anthropic_api_key = updates["ANTHROPIC_API_KEY"]
    if not updates:
        raise HTTPException(400, "Нечего сохранять")

    try:
        _persist_env(updates)
    except OSError as e:
        raise HTTPException(500, f"Не удалось записать .env: {e}") from e

    set_provider_keys(updates)
    return {"ok": True, "saved": list(updates.keys())}


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
