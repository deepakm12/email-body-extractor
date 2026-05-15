from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config.settings import LlmProviderType
from app.main import app
from app.providers.base import BaseLLMProvider
from app.services.history_service import HistoryRepository
from tests.samples import load_sample, load_sample_bytes


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    """Return a synchronous FastAPI test client."""
    return TestClient(app)


@pytest.fixture()
def mock_provider() -> MagicMock:
    """Return a fully-configured mock LLM provider."""
    provider = MagicMock(spec=BaseLLMProvider)
    provider.is_configured.return_value = True
    provider.invoke.return_value = '{"extracted_message": "Hello, this is the latest message."}'
    provider.stream.return_value = iter(["Hello, ", "this is ", "the latest."])
    provider.get_metadata.return_value = {
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "temperature": 0.0,
    }
    provider.provider_name = LlmProviderType.OPENAI
    provider.model_name = "gpt-4.1-mini"
    provider.temperature = 0.0
    return provider


@pytest.fixture()
def tmp_history_file(tmp_path: Path):  # type: ignore[type-arg]
    """Yield a HistoryRepository backed by a temporary file."""
    history_file = tmp_path / "history.json"
    yield HistoryRepository(file_path=history_file)


# ---------------------------------------------------------------------------
# Plain-text email fixtures  (backed by tests/samples/plain_text/)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_plain_email() -> str:
    """Plain-text reply email (Gmail-style On…wrote: block)."""
    return load_sample("plain_text/gmail_reply.txt")


@pytest.fixture()
def sample_email_with_signature() -> str:
    """Plain-text email followed by a Best-regards signature block."""
    return load_sample("plain_text/with_signature_closing.txt")


@pytest.fixture()
def sample_email_with_disclaimer() -> str:
    """Plain-text email followed by a CONFIDENTIALITY NOTICE block."""
    return load_sample("plain_text/with_confidentiality_notice.txt")


@pytest.fixture()
def sample_simple_message() -> str:
    """Clean single-message email with no threading or signature noise."""
    return load_sample("plain_text/simple_message.txt")


@pytest.fixture()
def sample_outlook_reply() -> str:
    """Plain-text email with Outlook From/Sent/To/Subject header block."""
    return load_sample("plain_text/outlook_reply.txt")


@pytest.fixture()
def sample_apple_mail_reply() -> str:
    """Plain-text email with Apple Mail 'On … at … wrote:' pattern."""
    return load_sample("plain_text/apple_mail_reply.txt")


@pytest.fixture()
def sample_multi_level_reply() -> str:
    """Three-level nested reply chain (Gmail-style)."""
    return load_sample("plain_text/multi_level_reply.txt")


@pytest.fixture()
def sample_signature_dash() -> str:
    """Email with a '--' separator followed by a corporate signature block."""
    return load_sample("plain_text/with_signature_dash.txt")


@pytest.fixture()
def sample_sent_from_device() -> str:
    """One-liner reply ending with 'Sent from my iPhone'."""
    return load_sample("plain_text/sent_from_device.txt")


@pytest.fixture()
def sample_full_thread() -> str:
    """Email combining a reply block, closing signature, and a disclaimer."""
    return load_sample("plain_text/full_thread.txt")


@pytest.fixture()
def sample_with_disclaimer() -> str:
    """Email followed by a DISCLAIMER: header block."""
    return load_sample("plain_text/with_disclaimer.txt")


# ---------------------------------------------------------------------------
# HTML email fixtures  (backed by tests/samples/html/)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_html_email() -> str:
    """HTML email with a Gmail div.gmail_quote quoted block."""
    return load_sample("html/gmail_quoted.html")


@pytest.fixture()
def sample_html_simple() -> str:
    """Plain HTML email with no quoted content."""
    return load_sample("html/simple.html")


@pytest.fixture()
def sample_html_outlook_forward() -> str:
    """HTML email with an Outlook-style forwarded-message block."""
    return load_sample("html/outlook_forward.html")


@pytest.fixture()
def sample_html_with_scripts() -> str:
    """HTML email containing script, style, and noscript tags."""
    return load_sample("html/with_scripts_styles.html")


# ---------------------------------------------------------------------------
# .eml fixtures  (backed by tests/samples/eml/)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_eml_plain() -> str:
    """Minimal .eml file with a text/plain body (decoded as str)."""
    return load_sample("eml/plain_text.eml")


@pytest.fixture()
def sample_eml_multipart() -> str:
    """Multipart .eml file with both text/plain and text/html parts."""
    return load_sample("eml/multipart.eml")


@pytest.fixture()
def sample_eml_html_only() -> str:
    """Single-part .eml file with a text/html body."""
    return load_sample("eml/html_only.eml")


# ---------------------------------------------------------------------------
# Edge-case fixtures  (backed by tests/samples/edge_cases/)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_ultra_short() -> str:
    """Two-character email body ('OK\\n')."""
    return load_sample("edge_cases/ultra_short.txt")


@pytest.fixture()
def sample_only_signature() -> str:
    """Content that consists entirely of a signature block with no message."""
    return load_sample("edge_cases/only_signature.txt")


@pytest.fixture()
def sample_unicode_content() -> str:
    """Non-ASCII email content (French with accented characters)."""
    return load_sample("edge_cases/unicode_content.txt")


@pytest.fixture()
def sample_long_email() -> str:
    """Multi-paragraph email well above the 10 000-char noise threshold."""
    return load_sample("edge_cases/long_email.txt")
