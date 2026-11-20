"""Единый интерфейс провайдера моделей.

Любой новый провайдер (локальный или API) реализует этот базовый класс
и регистрируется в `providers.registry`.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ModelInfo:
    name: str
    provider: str
    kind: str                    # local | api
    can_train: bool = False
    base_model: str = ""
    endpoint: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class ChatMessage:
    role: str                    # system | user | assistant
    content: str


class ProviderError(Exception):
    """Ошибка провайдера (сеть, ключ, неверная модель)."""


class ModelProvider(ABC):
    """Абстрактный провайдер: список моделей, инференс, обучение, health."""

    name: str = "base"
    kind: str = "api"

    @abstractmethod
    def list_models(self) -> list[ModelInfo]:
        ...

    @abstractmethod
    def run(self, model: str, messages: list[ChatMessage], max_tokens: int = 2048) -> str:
        ...

    def can_train(self, model: str) -> bool:
        return False

    def train(self, model: str, dataset_path: str, on_log=None, on_progress=None) -> dict:
        raise ProviderError(f"Провайдер {self.name} не поддерживает обучение")

    @abstractmethod
    def health(self) -> bool:
        ...
