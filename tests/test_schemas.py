from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.schemas import (
    ExtractionMode,
    ExtractionRequest,
    ExtractionResponse,
    ExtractionResult,
    HealthResponse,
    HealthStatus,
    ProviderInfo,
    ProvidersResponse,
)


class TestExtractionRequest:
    """Tests for ExtractionRequest validation."""

    def test_valid_with_defaults(self) -> None:
        req = ExtractionRequest(content="Hello world")
        assert req.content == "Hello world"
        assert req.mode == ExtractionMode.AUTO
        assert req.provider is None
        assert req.is_eml is False

    def test_whitespace_only_content_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExtractionRequest(content="   ")

    def test_empty_content_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExtractionRequest(content="")

    def test_all_extraction_modes_accepted(self) -> None:
        for mode in ExtractionMode:
            req = ExtractionRequest(content="Some email content", mode=mode)
            assert req.mode == mode

    def test_provider_field_is_optional(self) -> None:
        req = ExtractionRequest(content="Some email content", provider="openai")
        assert req.provider == "openai"

        req_no_provider = ExtractionRequest(content="Some email content")
        assert req_no_provider.provider is None

    def test_is_eml_flag(self) -> None:
        req = ExtractionRequest(content="MIME-Version: 1.0\n\nBody", is_eml=True)
        assert req.is_eml is True


class TestExtractionResult:
    """Tests for ExtractionResult validation."""

    def test_valid_result(self) -> None:
        result = ExtractionResult(
            latest_message="Hello world",
            confidence=0.9,
            flow_used="non_llm",
        )
        assert result.latest_message == "Hello world"
        assert result.confidence == 0.9
        assert result.flow_used == "non_llm"
        assert result.metadata == {}

    def test_confidence_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExtractionResult(
                latest_message="Hello",
                confidence=-0.1,
                flow_used="non_llm",
            )

    def test_confidence_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExtractionResult(
                latest_message="Hello",
                confidence=1.1,
                flow_used="non_llm",
            )

    def test_metadata_defaults_to_empty_dict(self) -> None:
        result = ExtractionResult(
            latest_message="Hello",
            confidence=0.5,
            flow_used="llm",
        )
        assert result.metadata == {}

    def test_metadata_accepts_dict(self) -> None:
        result = ExtractionResult(
            latest_message="Hello",
            confidence=0.75,
            flow_used="non_llm",
            metadata={"key": "value", "count": 42},
        )
        assert result.metadata["key"] == "value"
        assert result.metadata["count"] == 42


class TestExtractionResponse:
    """Tests for ExtractionResponse model."""

    def test_success_with_data(self) -> None:
        data = ExtractionResult(
            latest_message="Extracted text",
            confidence=0.95,
            flow_used="non_llm",
        )
        response = ExtractionResponse(success=True, data=data)
        assert response.success is True
        assert response.data is not None
        assert response.data.latest_message == "Extracted text"
        assert response.error is None

    def test_failure_with_error(self) -> None:
        response = ExtractionResponse(success=False, error="Something went wrong")
        assert response.success is False
        assert response.data is None
        assert response.error == "Something went wrong"

    def test_agent_trace_optional(self) -> None:
        response = ExtractionResponse(success=True)
        assert response.agent_trace is None

        trace = [{"agent": "extraction", "success": True}]
        response_with_trace = ExtractionResponse(success=True, agent_trace=trace)
        assert response_with_trace.agent_trace is not None
        assert len(response_with_trace.agent_trace) == 1


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_basic_construction(self) -> None:
        response = HealthResponse(version="1.0.0", status=HealthStatus.HEALTHY)
        assert response.version == "1.0.0"
        assert response.status == HealthStatus.HEALTHY

    def test_unhealthy_status(self) -> None:
        response = HealthResponse(version="dev", status=HealthStatus.UNHEALTHY)
        assert response.status == HealthStatus.UNHEALTHY


class TestProviderInfo:
    """Tests for ProviderInfo model."""

    def test_basic_construction(self) -> None:
        info = ProviderInfo(name="openai", available=True, configured=True)
        assert info.name == "openai"
        assert info.available is True
        assert info.configured is True

    def test_unconfigured_provider(self) -> None:
        info = ProviderInfo(name="anthropic", available=False, configured=False)
        assert info.available is False
        assert info.configured is False


class TestProvidersResponse:
    """Tests for ProvidersResponse model."""

    def test_basic_construction(self) -> None:
        providers = [
            ProviderInfo(name="openai", available=True, configured=True),
            ProviderInfo(name="anthropic", available=False, configured=False),
        ]
        response = ProvidersResponse(providers=providers, default_provider="openai")
        assert len(response.providers) == 2
        assert response.default_provider == "openai"

    def test_empty_providers_list(self) -> None:
        response = ProvidersResponse(providers=[], default_provider="none")
        assert response.providers == []
        assert response.default_provider == "none"
