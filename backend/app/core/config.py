"""Конфигурация платформы (env-переменные и пути)."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ -> корень проекта
ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    app_name: str = "FinAuditAI"
    database_url: str = f"sqlite:///{ROOT_DIR / 'data' / 'finauditai.db'}"

    # Локальные модели
    ollama_base_url: str = "http://localhost:11434"
    hf_home: str = str(ROOT_DIR / "data" / "models")

    # API-провайдеры (ключи только из env)
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str = ""

    # Проверщик
    finrepgym_dir: str = str(ROOT_DIR / "data" / "finrepgym")
    finrepgym_repo: str = "https://github.com/alivenoviop-afk/FinRepGYM"

    # Пути
    data_dir: Path = ROOT_DIR / "data"
    uploads_dir: Path = ROOT_DIR / "data" / "uploads"
    datasets_dir: Path = ROOT_DIR / "data" / "datasets"
    checkpoints_dir: Path = ROOT_DIR / "data" / "checkpoints"


settings = Settings()
for _p in (settings.data_dir, settings.uploads_dir, settings.datasets_dir, settings.checkpoints_dir):
    _p.mkdir(parents=True, exist_ok=True)
