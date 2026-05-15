from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.llm_flow.agents import (
    AgentResult,
    CleanupAgent,
    ConfidenceAgent,
    ExtractionAgent,
    ValidationAgent,
    cleanup_agent,
    confidence_agent,
)
from app.providers.base import BaseLLMProvider


# ---------------------------------------------------------------------------
# ExtractionAgent
# ---------------------------------------------------------------------------


class TestExtractionAgent:
    """Tests for ExtractionAgent.run()."""

    def test_run_returns_agent_result(self, mock_provider: MagicMock) -> None:
        agent = ExtractionAgent()
        result = agent.run(mock_provider, "Email content here.")
        assert isinstance(result, AgentResult)

    def test_agent_name_is_extraction(self) -> None:
        assert ExtractionAgent.agent_name == "extraction"

    def test_valid_json_response_extracts_message(self, mock_provider: MagicMock) -> None:
        mock_provider.invoke.return_value = '{"extracted_message": "Hello, this is the latest message."}'
        agent = ExtractionAgent()
        result = agent.run(mock_provider, "Some email thread")
        assert result.output == "Hello, this is the latest message."
        assert result.success is True

    def test_malformed_json_falls_back_to_raw_text(self, mock_provider: MagicMock) -> None:
        raw_response = "This is the extracted content (not JSON)."
        mock_provider.invoke.return_value = raw_response
        agent = ExtractionAgent()
        result = agent.run(mock_provider, "Email content")
        # Should fall back to using the raw text
        assert result.output == raw_response.strip()
        assert result.success is True

    def test_provider_exception_returns_failure_result(self, mock_provider: MagicMock) -> None:
        mock_provider.invoke.side_effect = RuntimeError("API error")
        agent = ExtractionAgent()
        result = agent.run(mock_provider, "Email content")
        assert result.success is False
        assert "error" in result.metadata


# ---------------------------------------------------------------------------
# CleanupAgent
# ---------------------------------------------------------------------------


class TestCleanupAgent:
    """Tests for CleanupAgent.run()."""

    def test_run_returns_agent_result(self, mock_provider: MagicMock) -> None:
        mock_provider.invoke.return_value = '{"cleaned_message": "Clean content."}'
        agent = CleanupAgent()
        result = agent.run(mock_provider, "Dirty content.")
        assert isinstance(result, AgentResult)

    def test_agent_name_is_cleanup(self) -> None:
        assert CleanupAgent.agent_name == "cleanup"

    def test_valid_json_extracts_cleaned_message(self, mock_provider: MagicMock) -> None:
        mock_provider.invoke.return_value = '{"cleaned_message": "Clean message here."}'
        agent = CleanupAgent()
        result = agent.run(mock_provider, "Noisy content <html>")
        assert result.output == "Clean message here."
        assert result.success is True

    def test_malformed_json_falls_back_to_raw_text(self, mock_provider: MagicMock) -> None:
        raw_response = "This is already clean text."
        mock_provider.invoke.return_value = raw_response
        agent = CleanupAgent()
        result = agent.run(mock_provider, "Some content")
        assert result.output == raw_response.strip()
        assert result.success is True

    def test_provider_exception_returns_failure_result(self, mock_provider: MagicMock) -> None:
        mock_provider.invoke.side_effect = Exception("Network error")
        agent = CleanupAgent()
        result = agent.run(mock_provider, "Content")
        assert result.success is False


# ---------------------------------------------------------------------------
# ValidationAgent
# ---------------------------------------------------------------------------


class TestValidationAgent:
    """Tests for ValidationAgent.run()."""

    def test_valid_json_true_result_success(self, mock_provider: MagicMock) -> None:
        mock_provider.invoke.return_value = '{"is_valid": true, "issues": [], "suggested_fix": ""}'
        agent = ValidationAgent()
        result = agent.run(mock_provider, "Clean extracted message.")
        assert result.success is True

    def test_valid_json_false_result_failure(self, mock_provider: MagicMock) -> None:
        mock_provider.invoke.return_value = '{"is_valid": false, "issues": ["Contains quoted reply"], "suggested_fix": "Remove quoted section"}'
        agent = ValidationAgent()
        result = agent.run(mock_provider, "Message with > quoted line")
        assert result.success is False

    def test_agent_name_is_validation(self) -> None:
        assert ValidationAgent.agent_name == "validation"

    def test_validation_metadata_contains_validation_key(self, mock_provider: MagicMock) -> None:
        mock_provider.invoke.return_value = '{"is_valid": true, "issues": [], "suggested_fix": ""}'
        agent = ValidationAgent()
        result = agent.run(mock_provider, "Good message.")
        assert "validation" in result.metadata

    def test_provider_exception_returns_success_true(self, mock_provider: MagicMock) -> None:
        # ValidationAgent._on_exception returns success=True to allow pipeline to continue
        mock_provider.invoke.side_effect = RuntimeError("Fail")
        agent = ValidationAgent()
        result = agent.run(mock_provider, "Content")
        assert result.success is True


# ---------------------------------------------------------------------------
# ConfidenceAgent
# ---------------------------------------------------------------------------


class TestConfidenceAgent:
    """Tests for ConfidenceAgent.run()."""

    def test_valid_json_extracts_confidence_score(self, mock_provider: MagicMock) -> None:
        mock_provider.invoke.return_value = '{"confidence_score": 0.9}'
        agent = ConfidenceAgent()
        result = agent.run(mock_provider, "Well-extracted message.")
        assert result.metadata.get("confidence_score") == pytest.approx(0.9)

    def test_agent_name_is_confidence(self) -> None:
        assert ConfidenceAgent.agent_name == "confidence"

    def test_non_json_numeric_string_extracts_value(self, mock_provider: MagicMock) -> None:
        mock_provider.invoke.return_value = "0.85"
        agent = ConfidenceAgent()
        result = agent.run(mock_provider, "Good message")
        score = result.metadata.get("confidence_score")
        assert score is not None
        assert 0.0 <= float(score) <= 1.0

    def test_provider_exception_returns_default_score(self, mock_provider: MagicMock) -> None:
        mock_provider.invoke.side_effect = RuntimeError("Error")
        agent = ConfidenceAgent()
        result = agent.run(mock_provider, "Content")
        assert result.success is True
        assert result.metadata.get("confidence_score") == pytest.approx(0.5)

    def test_result_output_is_original_content(self, mock_provider: MagicMock) -> None:
        mock_provider.invoke.return_value = '{"confidence_score": 0.8}'
        agent = ConfidenceAgent()
        original = "Original email message."
        result = agent.run(mock_provider, original)
        assert result.output == original


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


class TestModuleLevelAgentFunctions:
    """Tests for cleanup_agent() and confidence_agent() module functions."""

    def test_cleanup_agent_returns_agent_result(self, mock_provider: MagicMock) -> None:
        mock_provider.invoke.return_value = '{"cleaned_message": "Cleaned."}'
        result = cleanup_agent(mock_provider, "Some noisy email content")
        assert isinstance(result, AgentResult)

    def test_confidence_agent_returns_agent_result(self, mock_provider: MagicMock) -> None:
        mock_provider.invoke.return_value = '{"confidence_score": 0.9}'
        result = confidence_agent(mock_provider, "Clean message")
        assert isinstance(result, AgentResult)

    def test_cleanup_agent_agent_name(self, mock_provider: MagicMock) -> None:
        mock_provider.invoke.return_value = '{"cleaned_message": "Done."}'
        result = cleanup_agent(mock_provider, "Content")
        assert result.agent_name == "cleanup"

    def test_confidence_agent_agent_name(self, mock_provider: MagicMock) -> None:
        mock_provider.invoke.return_value = '{"confidence_score": 0.75}'
        result = confidence_agent(mock_provider, "Content")
        assert result.agent_name == "confidence"
