from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.models.schemas import ExtractionResponse, ExtractionResult


# ---------------------------------------------------------------------------
# GET /api/v1/health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for GET /api/v1/health."""

    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_returns_healthy_status(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_returns_version_key(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        data = response.json()
        assert "version" in data


# ---------------------------------------------------------------------------
# POST /api/v1/extract
# ---------------------------------------------------------------------------


class TestExtractEndpoint:
    """Tests for POST /api/v1/extract."""

    def test_empty_content_returns_422(self, client: TestClient) -> None:
        response = client.post("/api/v1/extract", json={"content": ""})
        assert response.status_code == 422

    def test_whitespace_only_content_returns_422(self, client: TestClient) -> None:
        response = client.post("/api/v1/extract", json={"content": "   "})
        assert response.status_code == 422

    def test_valid_content_non_llm_mode_returns_200(self, client: TestClient) -> None:
        """Non-LLM extraction requires no external API call."""
        payload = {
            "content": (
                "Hi Alice,\n\n"
                "Thanks for your update. I will send the report by Friday.\n\n"
                "Best regards,\nBob"
            ),
            "mode": "non_llm",
        }
        response = client.post("/api/v1/extract", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"] is not None
        assert data["data"]["latest_message"] != ""

    def test_successful_response_contains_required_fields(self, client: TestClient) -> None:
        payload = {
            "content": "Hello, this is a standalone email message.",
            "mode": "non_llm",
        }
        response = client.post("/api/v1/extract", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "data" in data
        result_data = data["data"]
        assert "latest_message" in result_data
        assert "confidence" in result_data
        assert "flow_used" in result_data

    def test_extraction_service_failure_returns_422(self, client: TestClient) -> None:
        """When ExtractionService.extract returns success=False the route raises 422."""
        failed_response = ExtractionResponse(success=False, error="Extraction failed")

        with patch(
            "app.api.v1.routes._EXTRACTION_SERVICE.extract",
            return_value=failed_response,
        ):
            response = client.post(
                "/api/v1/extract",
                json={"content": "Some email content", "mode": "non_llm"},
            )
        assert response.status_code == 422

    def test_missing_content_field_returns_422(self, client: TestClient) -> None:
        response = client.post("/api/v1/extract", json={"mode": "non_llm"})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/providers
# ---------------------------------------------------------------------------


class TestProvidersEndpoint:
    """Tests for GET /api/v1/providers."""

    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/providers")
        assert response.status_code == 200

    def test_returns_providers_list(self, client: TestClient) -> None:
        response = client.get("/api/v1/providers")
        data = response.json()
        assert "providers" in data
        assert isinstance(data["providers"], list)

    def test_returns_default_provider_key(self, client: TestClient) -> None:
        response = client.get("/api/v1/providers")
        data = response.json()
        assert "default_provider" in data

    def test_providers_list_has_expected_structure(self, client: TestClient) -> None:
        response = client.get("/api/v1/providers")
        data = response.json()
        for provider in data["providers"]:
            assert "name" in provider
            assert "available" in provider
            assert "configured" in provider


# ---------------------------------------------------------------------------
# GET /api/v1/history
# ---------------------------------------------------------------------------


class TestHistoryEndpoint:
    """Tests for GET /api/v1/history."""

    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/history")
        assert response.status_code == 200

    def test_returns_list(self, client: TestClient) -> None:
        response = client.get("/api/v1/history")
        data = response.json()
        assert isinstance(data, list)


# ---------------------------------------------------------------------------
# DELETE /api/v1/history
# ---------------------------------------------------------------------------


class TestDeleteHistoryEndpoint:
    """Tests for DELETE /api/v1/history."""

    def test_returns_204(self, client: TestClient) -> None:
        response = client.delete("/api/v1/history")
        assert response.status_code == 204

    def test_response_has_no_body(self, client: TestClient) -> None:
        response = client.delete("/api/v1/history")
        assert response.content == b""


# ---------------------------------------------------------------------------
# POST /api/v1/extract/stream
# ---------------------------------------------------------------------------


class TestExtractStreamEndpoint:
    """Tests for POST /api/v1/extract/stream."""

    def test_returns_200(self, client: TestClient) -> None:
        payload = {
            "content": "Hello, this is a streaming test email.",
            "mode": "non_llm",
        }
        response = client.post("/api/v1/extract/stream", json=payload)
        assert response.status_code == 200

    def test_content_type_is_event_stream(self, client: TestClient) -> None:
        payload = {
            "content": "Hello, this is a streaming test email.",
            "mode": "non_llm",
        }
        response = client.post("/api/v1/extract/stream", json=payload)
        assert "text/event-stream" in response.headers.get("content-type", "")

    def test_empty_content_returns_422(self, client: TestClient) -> None:
        response = client.post("/api/v1/extract/stream", json={"content": "", "mode": "non_llm"})
        assert response.status_code == 422

    def test_stream_response_contains_sse_data(self, client: TestClient) -> None:
        payload = {
            "content": (
                "Hi team,\n\nPlease review the attached report and provide feedback.\n\nThanks"
            ),
            "mode": "non_llm",
        }
        response = client.post("/api/v1/extract/stream", json=payload)
        assert response.status_code == 200
        body = response.text
        assert "data:" in body
