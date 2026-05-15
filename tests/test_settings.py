from __future__ import annotations

import pytest

from app.config.settings import AppSettings, LlmProviderType, get_settings


class TestAppSettingsDefaults:
    """Tests for default AppSettings values."""

    def test_app_name_default(self) -> None:
        settings = AppSettings()
        assert settings.app_name == "Email Body Extractor"

    def test_debug_default_false(self) -> None:
        settings = AppSettings()
        assert settings.debug is False

    def test_llm_provider_default_openai(self) -> None:
        settings = AppSettings()
        assert settings.llm_provider == LlmProviderType.OPENAI

    def test_confidence_threshold_default(self) -> None:
        settings = AppSettings()
        assert settings.confidence_threshold == pytest.approx(0.85)

    def test_temperature_default(self) -> None:
        settings = AppSettings()
        assert settings.temperature == pytest.approx(0.0)

    def test_model_name_default(self) -> None:
        settings = AppSettings()
        assert settings.model_name == "gpt-4.1-mini"


class TestCorsOriginsList:
    """Tests for the cors_origins_list property."""

    def test_single_origin(self) -> None:
        settings = AppSettings(cors_origins="https://example.com")
        assert settings.cors_origins_list == ["https://example.com"]

    def test_multiple_origins(self) -> None:
        settings = AppSettings(cors_origins="https://a.com,https://b.com,https://c.com")
        assert settings.cors_origins_list == ["https://a.com", "https://b.com", "https://c.com"]

    def test_wildcard_origin(self) -> None:
        settings = AppSettings(cors_origins="*")
        assert settings.cors_origins_list == ["*"]

    def test_origins_with_spaces(self) -> None:
        settings = AppSettings(cors_origins="https://a.com , https://b.com")
        assert settings.cors_origins_list == ["https://a.com", "https://b.com"]

    def test_empty_cors_origins_returns_empty_list(self) -> None:
        settings = AppSettings(cors_origins="")
        assert settings.cors_origins_list == []


class TestProviderApiKey:
    """Tests for the provider_api_key property."""

    def test_openai_key_returned_for_openai_provider(self) -> None:
        from pydantic import SecretStr

        settings = AppSettings(
            llm_provider=LlmProviderType.OPENAI,
            openai_api_key=SecretStr("sk-test-openai"),
        )
        key = settings.provider_api_key
        assert key is not None
        assert key.get_secret_value() == "sk-test-openai"

    def test_anthropic_key_returned_for_anthropic_provider(self) -> None:
        from pydantic import SecretStr

        settings = AppSettings(
            llm_provider=LlmProviderType.ANTHROPIC,
            anthropic_api_key=SecretStr("sk-ant-test"),
        )
        key = settings.provider_api_key
        assert key is not None
        assert key.get_secret_value() == "sk-ant-test"

    def test_gemini_key_returned_for_gemini_provider(self) -> None:
        from pydantic import SecretStr

        settings = AppSettings(
            llm_provider=LlmProviderType.GEMINI,
            gemini_api_key=SecretStr("gemini-key"),
        )
        key = settings.provider_api_key
        assert key is not None
        assert key.get_secret_value() == "gemini-key"

    def test_azure_key_returned_for_azure_provider(self) -> None:
        from pydantic import SecretStr

        settings = AppSettings(
            llm_provider=LlmProviderType.AZURE_OPENAI,
            azure_openai_api_key=SecretStr("azure-key"),
        )
        key = settings.provider_api_key
        assert key is not None
        assert key.get_secret_value() == "azure-key"

    def test_no_key_configured_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Ensure no OPENAI_API_KEY is set in the environment for this test
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings = AppSettings(llm_provider=LlmProviderType.OPENAI)
        assert settings.provider_api_key is None


class TestGetSettings:
    """Tests for the get_settings() cached factory function."""

    def test_returns_app_settings_instance(self) -> None:
        settings = get_settings()
        assert isinstance(settings, AppSettings)

    def test_cached_returns_same_object(self) -> None:
        first = get_settings()
        second = get_settings()
        assert first is second
