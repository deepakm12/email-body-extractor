"""Provider factory for runtime LLM provider selection."""

from app.config.settings import AppSettings, LlmProviderType, get_settings
from app.models.schemas import ProviderInfo
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.azure_openai_provider import AzureOpenAIProvider
from app.providers.base import BaseLLMProvider
from app.providers.gemini_provider import GeminiProvider
from app.providers.openai_provider import OpenAIProvider
from app.common.exceptions import ProviderNotConfiguredError
from app.common.logging_config import logger

# Default model names per provider
_DEFAULT_MODELS: dict[LlmProviderType, str] = {
    LlmProviderType.OPENAI: "gpt-4.1-mini",
    LlmProviderType.AZURE_OPENAI: "gpt-4.1-mini",
    LlmProviderType.ANTHROPIC: "claude-haiku-4-5",
    LlmProviderType.GEMINI: "gemini-2.5-flash-lite",
}


def _resolve_provider_name(provider_name: str) -> LlmProviderType:
    """Resolve a string provider name to a LlmProviderType."""
    normalized = provider_name.lower().replace("-", "_")
    match normalized:
        case LlmProviderType.OPENAI:
            return LlmProviderType.OPENAI
        case LlmProviderType.AZURE_OPENAI:
            return LlmProviderType.AZURE_OPENAI
        case LlmProviderType.ANTHROPIC:
            return LlmProviderType.ANTHROPIC
        case LlmProviderType.GEMINI:
            return LlmProviderType.GEMINI
        case _:
            raise ValueError(f"Unknown provider name: {provider_name}")


def _create_provider_instance(resolved_provider: LlmProviderType, settings: AppSettings) -> BaseLLMProvider:
    """Instantiate the correct provider class via match."""
    temperature = settings.temperature
    model_name = _DEFAULT_MODELS.get(resolved_provider, "")
    if resolved_provider == settings.llm_provider:
        logger.info(f"Using default provider from settings: {resolved_provider.value}")
        model_name = settings.model_name or model_name
    match resolved_provider:
        case LlmProviderType.OPENAI:
            return OpenAIProvider(
                model_name=model_name,
                temperature=temperature,
                api_key=settings.openai_api_key,
            )
        case LlmProviderType.AZURE_OPENAI:
            return AzureOpenAIProvider(
                model_name=settings.azure_openai_deployment or model_name,
                temperature=temperature,
                api_key=settings.azure_openai_api_key,
                endpoint=settings.azure_openai_endpoint,
            )
        case LlmProviderType.ANTHROPIC:
            return AnthropicProvider(
                model_name=model_name,
                temperature=temperature,
                api_key=settings.anthropic_api_key,
            )
        case LlmProviderType.GEMINI:
            return GeminiProvider(
                model_name=model_name,
                temperature=temperature,
                api_key=settings.gemini_api_key,
            )
        case _:
            raise ValueError(f"Unsupported provider type: {resolved_provider}")


def get_provider(provider_name: str | None, temperature: float | None = None) -> BaseLLMProvider:
    """Get an LLM provider instance."""
    settings = get_settings()
    resolved_provider = settings.llm_provider if provider_name is None else _resolve_provider_name(provider_name)
    temperature = temperature if temperature is not None else settings.temperature
    logger.info(
        "Creating LLM provider",
        extra={"provider": resolved_provider.value, "temperature": temperature},
    )
    try:
        instance = _create_provider_instance(resolved_provider=resolved_provider, settings=settings)
    except Exception as e:
        raise ProviderNotConfiguredError(f"Failed to initialize {resolved_provider.value} provider: {e}") from e
    if not instance.is_configured():
        raise ProviderNotConfiguredError(f"{resolved_provider.value} provider is not properly configured.")
    return instance


def list_available_providers() -> list[ProviderInfo]:
    """List all providers and their configuration status."""
    settings = get_settings()

    def _probe(provider: LlmProviderType) -> bool:
        try:
            instance = _create_provider_instance(resolved_provider=provider, settings=settings)
            return instance.is_configured()
        except Exception:
            return False

    return [
        ProviderInfo(name=provider.value, available=(status := _probe(provider)), configured=status)
        for provider in LlmProviderType
    ]
