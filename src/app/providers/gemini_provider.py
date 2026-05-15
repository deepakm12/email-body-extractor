"""Gemini LLM provider implementation."""

from collections.abc import Iterator

import openai
from openai import OpenAI
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from pydantic import SecretStr

from app.config.settings import LlmProviderType
from app.providers.base import BaseLLMProvider
from app.common.exceptions import ProviderError, ProviderNotConfiguredError, RetriableProviderError
from app.common.logging_config import logger

_RETRIABLE = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GeminiProvider(BaseLLMProvider):
    """Gemini LLM provider (via OpenAI-compatible API)."""

    _client: OpenAI | None
    _api_key: SecretStr | None

    def __init__(self, model_name: str, api_key: SecretStr | None = None, temperature: float = 0.0) -> None:
        super().__init__(model_name=model_name, temperature=temperature)
        self._api_key = api_key
        self._client = None
        if self._api_key is not None:
            self._client = OpenAI(
                api_key=self._api_key.get_secret_value(),
                base_url=GEMINI_BASE_URL,
            )

    @property
    def provider_name(self) -> LlmProviderType:
        """Name of the provider."""
        return LlmProviderType.GEMINI

    def is_configured(self) -> bool:
        """Check if Gemini API is configured."""
        return self._api_key is not None

    def invoke(self, prompt: str, system_message: str | None = None) -> str:
        """Invoke Gemini chat."""
        if not self.is_configured() or self._client is None:
            raise ProviderNotConfiguredError("Gemini provider not configured.")
        messages: list[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam] = []
        try:
            if system_message:
                messages.append(ChatCompletionSystemMessageParam(content=system_message, role="system", name="system"))
            messages.append(ChatCompletionUserMessageParam(content=prompt, role="user", name="user"))
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
            )
            result = response.choices[0].message.content or ""
            logger.debug(
                "Gemini invocation complete",
                extra={
                    "model": self.model_name,
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                },
            )
            return result
        except _RETRIABLE as e:
            logger.warning(f"Gemini transient error (will retry): {e}")
            raise RetriableProviderError(f"Gemini transient error: {e}") from e
        except Exception as e:
            logger.error(f"Gemini invocation failed: {e}")
            raise ProviderError(f"Gemini API error: {e}") from e

    def stream(self, prompt: str, system_message: str | None = None) -> Iterator[str]:
        """Stream Gemini chat completion tokens."""
        if not self.is_configured() or self._client is None:
            raise ProviderNotConfiguredError("Gemini provider not configured.")
        messages: list[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam] = []
        if system_message:
            messages.append(ChatCompletionSystemMessageParam(content=system_message, role="system", name="system"))
        messages.append(ChatCompletionUserMessageParam(content=prompt, role="user", name="user"))
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                stream=True,
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except _RETRIABLE as e:
            logger.warning(f"Gemini stream transient error (will retry): {e}")
            raise RetriableProviderError(f"Gemini stream transient error: {e}") from e
        except Exception as e:
            logger.error(f"Gemini stream failed: {e}")
            raise ProviderError(f"Gemini stream error: {e}") from e

    def get_metadata(self) -> dict[str, object]:
        metadata = super().get_metadata()
        metadata["provider_name"] = self.provider_name.value
        return metadata
