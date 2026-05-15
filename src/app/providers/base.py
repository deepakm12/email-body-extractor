"""Base LLM provider interface."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from app.config.settings import LlmProviderType
from app.common.logging_config import logger


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model_name: str, temperature: float = 0.0) -> None:
        self.logger = logger
        self.model_name = model_name
        self.temperature = temperature

    @property
    @abstractmethod
    def provider_name(self) -> LlmProviderType:
        """Name of the provider."""

    @abstractmethod
    def invoke(self, prompt: str, system_message: str | None = None) -> str:
        """Invoke the LLM with a prompt."""

    @abstractmethod
    def stream(self, prompt: str, system_message: str | None = None) -> Iterator[str]:
        """Stream response tokens. Yields text chunks as they arrive."""

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if the provider is properly configured."""

    def get_metadata(self) -> dict[str, object]:
        """Get provider metadata."""
        return {
            "provider": self.provider_name.value,
            "model": self.model_name,
            "temperature": self.temperature,
        }
