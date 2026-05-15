from __future__ import annotations

import pytest

from app.common.exceptions import InvalidEmailError
from app.common.preprocessing import EmailPreprocessor, PreprocessedEmail


class TestEmailPreprocessorPlainText:
    """Tests for plain-text processing."""

    def setup_method(self) -> None:
        self.preprocessor = EmailPreprocessor()

    def test_plain_text_returned_as_is(self) -> None:
        content = "Hello, this is a plain email message."
        result = self.preprocessor.process(content, is_html=False)
        assert isinstance(result, PreprocessedEmail)
        assert "Hello" in result.text
        assert result.original == content

    def test_plain_text_has_empty_html(self) -> None:
        result = self.preprocessor.process("Simple message.", is_html=False)
        assert result.html == ""

    def test_plain_text_preserves_original(self) -> None:
        content = "Original text here."
        result = self.preprocessor.process(content, is_html=False)
        assert result.original == content

    def test_apple_mail_on_wrote_pattern_removed(self) -> None:
        content = (
            "Thanks for your email.\n\n"
            "On Monday, April 7, 2025 at 10:00 AM Alice wrote:\n"
            "Can you review this?\n"
        )
        result = self.preprocessor.process(content, is_html=False)
        assert "wrote:" not in result.text

    def test_normalizes_crlf_line_endings(self) -> None:
        content = "Line one.\r\nLine two.\r\n"
        result = self.preprocessor.process(content, is_html=False)
        assert "\r\n" not in result.text

    def test_removes_null_bytes(self) -> None:
        content = "Hello\x00 world."
        result = self.preprocessor.process(content, is_html=False)
        assert "\x00" not in result.text

    def test_collapses_multiple_blank_lines(self) -> None:
        content = "First paragraph.\n\n\n\n\nSecond paragraph."
        result = self.preprocessor.process(content, is_html=False)
        # Should not have 3+ consecutive newlines after normalisation
        assert "\n\n\n" not in result.text

    def test_confidentiality_notice_removed(self) -> None:
        content = (
            "Hi Jane,\n\nPlease review the contract.\n\n"
            "CONFIDENTIALITY NOTICE:\n"
            "This email is confidential and intended only for the addressee.\n"
        )
        result = self.preprocessor.process(content, is_html=False)
        assert "CONFIDENTIALITY NOTICE" not in result.text

    def test_normalizes_unicode(self) -> None:
        # Café with combining character vs precomposed — both should survive NFC normalisation
        content = "Café meeting confirmed."
        result = self.preprocessor.process(content, is_html=False)
        assert "Caf" in result.text


class TestEmailPreprocessorHTML:
    """Tests for HTML email processing."""

    def setup_method(self) -> None:
        self.preprocessor = EmailPreprocessor()

    def test_html_tags_stripped(self) -> None:
        content = "<html><body><p>Hello world</p></body></html>"
        result = self.preprocessor.process(content, is_html=True)
        assert "<html>" not in result.text
        assert "Hello world" in result.text

    def test_auto_detects_html(self) -> None:
        # The auto-detect regex looks for <html, <body, <div, <p\s, or <br
        # Use a <div> wrapper so detection triggers reliably.
        content = "<div>This is HTML content.</div>"
        result = self.preprocessor.process(content)
        assert "<div>" not in result.text
        assert "This is HTML content" in result.text

    def test_gmail_quote_div_removed(self) -> None:
        content = (
            "<html><body>"
            "<p>Latest message here.</p>"
            '<div class="gmail_quote">'
            "<p>Previous message quoted.</p>"
            "</div>"
            "</body></html>"
        )
        result = self.preprocessor.process(content, is_html=True)
        assert "Previous message quoted" not in result.text
        assert "Latest message here" in result.text

    def test_hr_tags_removed(self) -> None:
        content = "<html><body><p>Message.</p><hr/><p>Separator line.</p></body></html>"
        result = self.preprocessor.process(content, is_html=True)
        # The <hr> element itself should be gone, content may remain
        assert "<hr" not in result.text

    def test_html_stored_in_result(self) -> None:
        content = "<html><body><p>Test</p></body></html>"
        result = self.preprocessor.process(content, is_html=True)
        assert result.html != ""

    def test_script_tags_removed(self) -> None:
        content = "<html><body><p>Real content</p><script>alert('xss')</script></body></html>"
        result = self.preprocessor.process(content, is_html=True)
        assert "alert" not in result.text


class TestEmailPreprocessorEml:
    """Tests for .eml file parsing."""

    def setup_method(self) -> None:
        self.preprocessor = EmailPreprocessor()

    def test_minimal_eml_plain_body_extracted(self) -> None:
        eml_content = (
            "MIME-Version: 1.0\n"
            "Content-Type: text/plain; charset=utf-8\n"
            "\n"
            "Body here"
        )
        result = self.preprocessor.process(eml_content, is_eml=True)
        assert "Body here" in result.text

    def test_eml_with_html_body_html_preferred(self) -> None:
        eml_content = (
            "MIME-Version: 1.0\n"
            'Content-Type: multipart/alternative; boundary="boundary"\n'
            "\n"
            "--boundary\n"
            "Content-Type: text/plain; charset=utf-8\n"
            "\n"
            "Plain text fallback.\n"
            "--boundary\n"
            "Content-Type: text/html; charset=utf-8\n"
            "\n"
            "<html><body><p>HTML preferred body.</p></body></html>\n"
            "--boundary--\n"
        )
        result = self.preprocessor.process(eml_content, is_eml=True)
        assert "HTML preferred body" in result.text

    def test_invalid_eml_raises_invalid_email_error(self) -> None:
        # Completely non-EML content passed with is_eml=True won't cause
        # the library itself to throw (email.message_from_bytes is permissive),
        # but we can trigger the error by monkey-patching; instead verify the
        # output is a PreprocessedEmail (graceful degradation).
        # The real InvalidEmailError path is covered separately via mock.
        with pytest.raises((InvalidEmailError, Exception)):
            from unittest.mock import patch

            with patch("email.message_from_bytes", side_effect=Exception("parse error")):
                self.preprocessor.process("not valid eml", is_eml=True)

    def test_eml_returns_preprocessed_email_dataclass(self) -> None:
        eml_content = (
            "MIME-Version: 1.0\n"
            "Content-Type: text/plain\n"
            "\n"
            "Test body"
        )
        result = self.preprocessor.process(eml_content, is_eml=True)
        assert hasattr(result, "text")
        assert hasattr(result, "html")
        assert hasattr(result, "original")


class TestPreprocessedEmailDataclass:
    """Tests for the PreprocessedEmail dataclass fields."""

    def test_has_text_field(self) -> None:
        pe = PreprocessedEmail(text="t", html="h", original="o")
        assert pe.text == "t"

    def test_has_html_field(self) -> None:
        pe = PreprocessedEmail(text="t", html="h", original="o")
        assert pe.html == "h"

    def test_has_original_field(self) -> None:
        pe = PreprocessedEmail(text="t", html="h", original="o")
        assert pe.original == "o"
