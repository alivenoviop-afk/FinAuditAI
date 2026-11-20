"""FinAuditAI — FastAPI: REST API + раздача веб-интерфейса."""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import inference as inference_api
from app.api import models as models_api
from app.api import training as training_api
from app.core.config import settings

app = FastAPI(title=settings.app_name, version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для продакшна — сузить до домена фронта
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models_api.router)
app.include_router(training_api.router)
app.include_router(inference_api.router)


@app.get("/api/health")
def health():
    from app.providers.registry import all_providers
    return {
        "app": settings.app_name,
        "providers": [{"name": p.name, "kind": p.kind, "health": p.health()}
                      for p in all_providers().values()],
    }


# Раздача веб-интерфейса: frontend/dist (сборка) или frontend/ (plain, без сборки)
_root = Path(__file__).resolve().parents[2]
_dist = _root / "frontend" / "dist"
_static_dir = _dist if _dist.exists() else _root / "frontend"

if (_static_dir / "index.html").exists():
    _assets = _static_dir / "assets"
    if _assets.exists():
        app.mount("/assets", StaticFiles(directory=_assets), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        # API-пути не перехватываем — иначе 404 на API превращается в index.html
        if full_path.startswith("api/"):
            raise HTTPException(404, "Not Found")
        candidate = _static_dir / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_static_dir / "index.html")
