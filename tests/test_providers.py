from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from app.common.exceptions import ProviderNotConfiguredError
from app.config.settings import LlmProviderType
from app.providers.azure_openai_provider import AzureOpenAIProvider, _build_azure_base_url
from app.providers.factory import _resolve_provider_name, get_provider
from app.providers.openai_provider import OpenAIProvider


# ---------------------------------------------------------------------------
# _build_azure_base_url
# ---------------------------------------------------------------------------


class TestBuildAzureBaseUrl:
    """Tests for the _build_azure_base_url helper."""

    def test_endpoint_ending_with_com_slash_gets_openai_v1_appended(self) -> None:
        endpoint = "https://my-resource.openai.azure.com/"
        result = _build_azure_base_url(endpoint)
        assert result == "https://my-resource.openai.azure.com/openai/v1/"

    def test_endpoint_already_containing_openai_v1_unchanged(self) -> None:
        endpoint = "https://my-resource.openai.azure.com/openai/v1/"
        result = _build_azure_base_url(endpoint)
        assert result == endpoint

    def test_endpoint_without_trailing_slash_unchanged(self) -> None:
        endpoint = "https://my-resource.openai.azure.com"
        result = _build_azure_base_url(endpoint)
        # Does not end with ".com/" so should be returned as-is
        assert result == endpoint


# ---------------------------------------------------------------------------
# _resolve_provider_name
# ---------------------------------------------------------------------------


class TestResolveProviderName:
    """Tests for the _resolve_provider_name factory helper."""

    def test_openai_string_resolves(self) -> None:
        assert _resolve_provider_name("openai") == LlmProviderType.OPENAI

    def test_azure_openai_underscore_resolves(self) -> None:
        assert _resolve_provider_name("azure_openai") == LlmProviderType.AZURE_OPENAI

    def test_azure_openai_hyphen_normalised(self) -> None:
        assert _resolve_provider_name("azure-openai") == LlmProviderType.AZURE_OPENAI

    def test_anthropic_string_resolves(self) -> None:
        assert _resolve_provider_name("anthropic") == LlmProviderType.ANTHROPIC

    def test_gemini_string_resolves(self) -> None:
        assert _resolve_provider_name("gemini") == LlmProviderType.GEMINI

    def test_unknown_provider_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            _resolve_provider_name("unknown_provider")


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    def test_is_configured_false_when_no_api_key(self) -> None:
        provider = OpenAIProvider(model_name="gpt-4.1-mini", api_key=None)
        assert provider.is_configured() is False

    def test_is_configured_true_when_api_key_provided(self) -> None:
        provider = OpenAIProvider(
            model_name="gpt-4.1-mini",
            api_key=SecretStr("sk-test-key"),
        )
        assert provider.is_configured() is True

    def test_invoke_raises_when_not_configured(self) -> None:
        provider = OpenAIProvider(model_name="gpt-4.1-mini", api_key=None)
        with pytest.raises(ProviderNotConfiguredError):
            provider.invoke("Hello")

    def test_stream_raises_when_not_configured(self) -> None:
        provider = OpenAIProvider(model_name="gpt-4.1-mini", api_key=None)
        with pytest.raises(ProviderNotConfiguredError):
            list(provider.stream("Hello"))

    def test_get_metadata_returns_dict_with_required_keys(self) -> None:
        provider = OpenAIProvider(model_name="gpt-4.1-mini", temperature=0.0, api_key=None)
        metadata = provider.get_metadata()
        assert "provider" in metadata
        assert "model" in metadata
        assert "temperature" in metadata

    def test_get_metadata_values(self) -> None:
        provider = OpenAIProvider(model_name="gpt-4.1-mini", temperature=0.2, api_key=None)
        metadata = provider.get_metadata()
        assert metadata["model"] == "gpt-4.1-mini"
        assert metadata["temperature"] == pytest.approx(0.2)

    def test_provider_name_returns_openai_enum(self) -> None:
        provider = OpenAIProvider(model_name="gpt-4.1-mini", api_key=None)
        assert provider.provider_name == LlmProviderType.OPENAI

    def test_invoke_calls_openai_client(self) -> None:
        """Verify invoke() delegates to the OpenAI client (mocked)."""
        provider = OpenAIProvider(
            model_name="gpt-4.1-mini",
            api_key=SecretStr("sk-fake"),
        )
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        with patch.object(provider._client.chat.completions, "create", return_value=mock_response):
            result = provider.invoke("Test prompt")
        assert result == "Test response"


# ---------------------------------------------------------------------------
# get_provider factory
# ---------------------------------------------------------------------------


class TestGetProvider:
    """Tests for the get_provider() factory function."""

    def test_raises_when_provider_api_key_not_set(self) -> None:
        """get_provider should raise ProviderNotConfiguredError when the API key is absent."""
        with patch("app.providers.factory.get_settings") as mock_settings:
            settings = MagicMock()
            settings.llm_provider = LlmProviderType.OPENAI
            settings.model_name = "gpt-4.1-mini"
            settings.temperature = 0.0
            settings.provider_api_key = None
            settings.azure_openai_endpoint = None
            settings.azure_openai_deployment = None
            mock_settings.return_value = settings

            with pytest.raises(ProviderNotConfiguredError):
                get_provider(provider_name=None)

    def test_uses_default_settings_when_provider_name_is_none(self) -> None:
        """get_provider(None) uses settings.llm_provider."""
        with patch("app.providers.factory.get_settings") as mock_settings:
            settings = MagicMock()
            settings.llm_provider = LlmProviderType.OPENAI
            settings.model_name = "gpt-4.1-mini"
            settings.temperature = 0.0
            settings.provider_api_key = None
            settings.azure_openai_endpoint = None
            settings.azure_openai_deployment = None
            mock_settings.return_value = settings

            # Without a configured key this raises; just check it attempts OPENAI
            with pytest.raises(ProviderNotConfiguredError) as exc_info:
                get_provider(provider_name=None)
            assert "openai" in str(exc_info.value).lower()
