"""Реестр провайдеров: единая точка доступа «выбрать и подключить модель»."""
from __future__ import annotations

from app.providers.api import AnthropicProvider, OpenAIProvider
from app.providers.base import ModelInfo, ModelProvider, ProviderError
from app.providers.local import HuggingFaceProvider, OllamaProvider

# Пользовательские API-эндпоинты, добавленные через UI: name -> provider
_custom: dict[str, ModelProvider] = {}

_builtin: dict[str, ModelProvider] = {
    "ollama": OllamaProvider(),
    "huggingface": HuggingFaceProvider(),
    "openai": OpenAIProvider(),
    "anthropic": AnthropicProvider(),
}


def get_provider(name: str) -> ModelProvider:
    if name in _custom:
        return _custom[name]
    if name in _builtin:
        return _builtin[name]
    raise ProviderError(f"Неизвестный провайдер: {name}")


def register_custom_api(name: str, base_url: str, api_key: str) -> ModelProvider:
    """Подключить свой API (OpenAI-совместимый) — ключ хранится только в памяти/env."""
    if not name or not base_url:
        raise ProviderError("Нужны name и base_url")
    prov = OpenAIProvider(api_key=api_key, base_url=base_url)
    prov.name = name
    _custom[name] = prov
    return prov


def all_providers() -> dict[str, ModelProvider]:
    return {**_builtin, **_custom}


def set_provider_keys(updates: dict[str, str]) -> None:
    """Применить сохранённые ключи к уже созданным провайдерам (без перезапуска)."""
    if "OPENAI_API_KEY" in updates:
        _builtin["openai"].api_key = updates["OPENAI_API_KEY"]
    if "OPENAI_BASE_URL" in updates:
        _builtin["openai"].base_url = updates["OPENAI_BASE_URL"].rstrip("/")
    if "ANTHROPIC_API_KEY" in updates:
        _builtin["anthropic"].api_key = updates["ANTHROPIC_API_KEY"]


def collect_models() -> tuple[list[ModelInfo], list[str]]:
    """Список всех моделей + ошибки недоступных провайдеров.

    Провайдеры без ключа / локальные, которые не запущены, — не считаются
    ошибкой: они молча пропускаются и помечаются статусом в /api/models.
    """
    models: list[ModelInfo] = []
    errors: list[str] = []
    for name, prov in all_providers().items():
        # API без ключа и локальные, которые не запущены — просто не настроены,
        # не шумим в errors: статус виден в /api/models (ok/no_key/offline)
        if prov.kind == "api" and not getattr(prov, "api_key", ""):
            continue
        try:
            models.extend(prov.list_models())
        except ProviderError:
            continue  # провайдер недоступен — статус покажет provider_status
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e}")
    return models, errors
