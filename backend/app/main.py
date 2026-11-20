"""FinAuditAI — FastAPI: REST API + раздача веб-интерфейса."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import eval as eval_api
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
app.include_router(eval_api.router)


@app.get("/api/health")
def health():
    from app.providers.registry import all_providers
    return {
        "app": settings.app_name,
        "providers": [{"name": p.name, "kind": p.kind, "health": p.health()}
                      for p in all_providers().values()],
    }


# Раздача собранного фронтенда (frontend/dist), если есть — иначе dev-страница
_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _dist.exists():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        candidate = _dist / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_dist / "index.html")
