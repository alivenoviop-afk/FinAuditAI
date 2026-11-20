"""API-провайдеры: OpenAI (и любой OpenAI-совместимый endpoint) + Anthropic Claude."""
from __future__ import annotations

import httpx

from app.core.config import settings
from app.providers.base import ChatMessage, ModelInfo, ModelProvider, ProviderError


class OpenAIProvider(ModelProvider):
    """OpenAI API и любой совместимый endpoint (внутренние LLM компании, vLLM, ...).

    Endpoint и ключ настраиваются в UI/через env: OPENAI_API_KEY, OPENAI_BASE_URL.
    """

    name = "openai"
    kind = "api"

    def __init__(self, api_key: str = "", base_url: str = ""):
        self.api_key = api_key or settings.openai_api_key
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")

    def _headers(self) -> dict:
        if not self.api_key:
            raise ProviderError("Не задан OPENAI_API_KEY (или ключ не добавлен в UI)")
        return {"Authorization": f"Bearer {self.api_key}"}

    def list_models(self) -> list[ModelInfo]:
        try:
            r = httpx.get(f"{self.base_url}/models", headers=self._headers(), timeout=15)
            r.raise_for_status()
        except Exception as e:
            raise ProviderError(f"Ошибка списка моделей: {e}") from e
        return [
            ModelInfo(name=m["id"], provider=self.name, kind=self.kind, can_train=False,
                      endpoint=self.base_url)
            for m in r.json().get("data", [])
        ]

    def run(self, model: str, messages: list[ChatMessage], max_tokens: int = 2048) -> str:
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
        }
        try:
            r = httpx.post(f"{self.base_url}/chat/completions", json=payload,
                           headers=self._headers(), timeout=120)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ProviderError(f"API {e.response.status_code}: {e.response.text[:300]}") from e
        except Exception as e:
            raise ProviderError(f"Ошибка запроса: {e}") from e
        return r.json()["choices"][0]["message"]["content"]

    def health(self) -> bool:
        try:
            return bool(self.list_models())
        except Exception:
            return False


class AnthropicProvider(ModelProvider):
    """Anthropic Claude (Messages API)."""

    name = "anthropic"
    kind = "api"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or settings.anthropic_api_key

    def _headers(self) -> dict:
        if not self.api_key:
            raise ProviderError("Не задан ANTHROPIC_API_KEY")
        return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}

    def list_models(self) -> list[ModelInfo]:
        try:
            r = httpx.get("https://api.anthropic.com/v1/models", headers=self._headers(), timeout=15)
            r.raise_for_status()
        except Exception as e:
            raise ProviderError(f"Ошибка списка моделей Anthropic: {e}") from e
        return [ModelInfo(name=m["id"], provider=self.name, kind=self.kind, endpoint="https://api.anthropic.com")
                for m in r.json().get("data", [])]

    def run(self, model: str, messages: list[ChatMessage], max_tokens: int = 2048) -> str:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        chat = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        payload = {"model": model, "messages": chat, "max_tokens": max_tokens}
        if system:
            payload["system"] = system
        try:
            r = httpx.post("https://api.anthropic.com/v1/messages", json=payload,
                           headers=self._headers(), timeout=120)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ProviderError(f"API {e.response.status_code}: {e.response.text[:300]}") from e
        except Exception as e:
            raise ProviderError(f"Ошибка запроса: {e}") from e
        blocks = r.json().get("content", [])
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")

    def health(self) -> bool:
        try:
            return bool(self.list_models())
        except Exception:
            return False
