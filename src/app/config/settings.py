"""Application configuration."""

import os
from enum import Enum
from functools import lru_cache
from importlib.metadata import version as pkg_version

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ENVFILE_PATH_KEY = "APPLICATION_ENVFILE_PATH"
API_V1_PREFIX = "/api/v1"


def _detect_version() -> str:
    """Read version from installed package metadata or pyproject.toml."""
    try:
        return pkg_version("email-body-extractor")
    except Exception:
        pass
    return "dev"


class LlmProviderType(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


class LogLevel(str, Enum):
    """Defines valid log levels for the app."""

    INFO = "INFO"
    DEBUG = "DEBUG"


class AppSettings(BaseSettings):
    """Application configuration settings."""

    model_config = SettingsConfigDict(env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    # App configuration
    app_name: str = "Email Body Extractor"
    app_description: str = (
        "Hybrid Email Body Extraction Platform - extracts the latest meaningful"
        " email message using deterministic NLP and agentic AI workflows."
    )
    app_version: str = _detect_version()
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    cors_origins: str = "*"

    # LLM Configuration
    llm_provider: LlmProviderType = LlmProviderType.OPENAI
    model_name: str = "gpt-4.1-mini"
    temperature: float = 0.0
    openai_api_key: SecretStr | None = None
    azure_openai_api_key: SecretStr | None = None
    azure_openai_endpoint: SecretStr | None = None
    azure_openai_deployment: str | None = None
    gemini_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    # Extraction Configuration
    confidence_threshold: float = 0.85

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def provider_api_key(self) -> SecretStr | None:
        """Get the API key for the configured provider."""
        match self.llm_provider:
            case LlmProviderType.OPENAI:
                return self.openai_api_key
            case LlmProviderType.AZURE_OPENAI:
                return self.azure_openai_api_key
            case LlmProviderType.GEMINI:
                return self.gemini_api_key
            case LlmProviderType.ANTHROPIC:
                return self.anthropic_api_key
            case _:
                raise ValueError(f"Unsupported provider: {self.llm_provider}")


@lru_cache()
def get_settings() -> AppSettings:
    """Get cached settings instance."""
    env_file = os.getenv(ENVFILE_PATH_KEY, ".env")
    return AppSettings(_env_file=env_file)  # type: ignore[call-arg]
