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


def collect_models() -> tuple[list[ModelInfo], list[str]]:
    """Список всех моделей + ошибки недоступных провайдеров."""
    models: list[ModelInfo] = []
    errors: list[str] = []
    for name, prov in all_providers().items():
        try:
            models.extend(prov.list_models())
        except ProviderError as e:
            errors.append(f"{name}: {e}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e}")
    return models, errors
