"""Shared preprocessing module for email content normalization."""

import email
import re
import unicodedata
from dataclasses import dataclass
from email.message import Message

from bs4 import BeautifulSoup, Comment

from app.common.exceptions import InvalidEmailError, PreprocessingError
from app.common.logging_config import logger


@dataclass
class PreprocessedEmail:
    """Result of the email preprocessing pipeline."""

    text: str
    html: str
    original: str


# Patterns for Gmail quoted content
GMAIL_QUOTE_SELECTORS = [
    "div.gmail_quote",
    "blockquote.gmail_quote",
    ".gmail_attr",
    ".gmail_extra",
]

# Patterns for Outlook forwarded content
OUTLOOK_PATTERNS = [
    r"From:\s*.*?\nSent:\s*.*?\nTo:\s*.*?\nSubject:\s*.*?\n",
    r"________________________________________\nFrom:.*?\nSent:.*?\nTo:.*?\nSubject:.*?\n",
    r"-{3,}Original Message-{3,}",
    r"Begin forwarded message:",
]

# Patterns for Apple Mail
APPLE_MAIL_PATTERNS = [
    r"On\s+.*?wrote:",
    r">\s*On\s+.*?wrote:",
]

# Common disclaimer patterns
DISCLAIMER_PATTERNS = [
    r"(?:^|\n)\s*CONFIDENTIALITY NOTICE\s*[:\n].*?(?=\n\n|\Z)",
    r"(?:^|\n)\s*DISCLAIMER\s*[:\n].*?(?=\n\n|\Z)",
    r"(?:^|\n)\s*This\s+(?:e-?mail|message)\s+is\s+confidential.*?(?=\n\n|\Z)",
    r"(?:^|\n)\s*If\s+you\s+have\s+received\s+this\s+(?:e-?mail|message)\s+in\s+error.*?(?=\n\n|\Z)",
    r"(?:^|\n)\s*The\s+information\s+contained.*?is\s+confidential.*?(?=\n\n|\Z)",
    r"(?:^|\n)\s*Please\s+consider\s+the\s+environment\s+before\s+printing",
]

# HTML tags to remove completely
REMOVE_TAGS = ["script", "style", "noscript", "iframe", "canvas", "svg"]


class EmailPreprocessor:
    """Preprocesses raw email content into normalized plain text."""

    def process(self, content: str, is_html: bool | None = None, is_eml: bool = False) -> PreprocessedEmail:
        """Main preprocessing pipeline for email content."""
        logger.info(
            "Starting email preprocessing",
            extra={"is_html": is_html, "is_eml": is_eml, "content_length": len(content)},
        )
        original = content
        html_body: str | None = None
        if is_eml:
            eml_bytes = content.encode("utf-8", errors="replace")
            text_body, html_body = self._parse_eml_file(eml_bytes)
            if html_body:
                is_html = True
                content = html_body
            else:
                is_html = False
                content = text_body or ""
        if is_html is None:
            is_html = bool(re.search(r"<html|<body|<div|<p\s|<br", content, re.I))
            logger.debug(f"Auto-detected content type: {'HTML' if is_html else 'plain text'}")
        if is_html:
            cleaned_html = self._clean_html(content)
            text = self._html_to_text(cleaned_html)
            html_body = cleaned_html
        else:
            text = content
            html_body = None
        text = self._remove_apple_mail_quotes(text)
        text = self._normalize_text(text)
        text = self._remove_disclaimers(text)
        logger.info(
            "Email preprocessing complete",
            extra={
                "original_length": len(original),
                "processed_length": len(text),
                "is_html": is_html,
            },
        )
        return PreprocessedEmail(text=text, html=html_body or "", original=original)

    def _parse_eml_file(self, eml_content: bytes) -> tuple[str, str | None]:
        """Parse an .eml file and extract body content."""
        try:
            msg = email.message_from_bytes(eml_content)
        except Exception as e:
            raise InvalidEmailError(f"Failed to parse .eml file: {e}") from e
        text_body: str | None = None
        html_body: str | None = None

        def extract_part(part: Message) -> None:
            nonlocal text_body, html_body
            content_type = part.get_content_type()
            payload = part.get_payload(decode=True)
            if payload is None:
                return
            charset = part.get_content_charset() or "utf-8"
            raw: bytes = payload if isinstance(payload, bytes) else str(payload).encode("utf-8")
            try:
                decoded = raw.decode(charset, errors="replace")
            except (LookupError, UnicodeDecodeError):
                decoded = raw.decode("utf-8", errors="replace")
            if content_type == "text/plain" and text_body is None:
                text_body = decoded
            elif content_type == "text/html" and html_body is None:
                html_body = decoded

        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_maintype() == "multipart":
                    continue
                extract_part(part)
        else:
            extract_part(msg)
        logger.debug(
            "Parsed .eml file",
            extra={
                "has_text_body": text_body is not None,
                "has_html_body": html_body is not None,
                "subject": msg.get("Subject", "Unknown"),
            },
        )
        return text_body or "", html_body

    def _remove_gmail_quotes(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Remove Gmail quoted content from BeautifulSoup object."""
        removed_any = False
        for selector in GMAIL_QUOTE_SELECTORS:
            for element in soup.select(selector):
                element.decompose()
                removed_any = True
        if removed_any:
            logger.debug("Removed Gmail quoted content")
        return soup

    def _remove_outlook_quotes(self, html_content: str) -> str:
        """Remove Outlook forwarded/quoted content."""
        for pattern in OUTLOOK_PATTERNS:
            html_content = re.sub(pattern, "", html_content, flags=re.IGNORECASE | re.DOTALL)
        return html_content

    def _remove_apple_mail_quotes(self, text_content: str) -> str:
        """Remove Apple Mail quoted content markers."""
        for pattern in APPLE_MAIL_PATTERNS:
            text_content = re.sub(pattern, "", text_content, flags=re.IGNORECASE | re.DOTALL)
        return text_content

    def _clean_html(self, html_content: str) -> str:
        """Clean HTML content by removing unwanted tags and elements."""
        try:
            html_content = self._remove_outlook_quotes(html_content)
            soup = BeautifulSoup(html_content, "lxml")
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
            for tag_name in REMOVE_TAGS:
                for tag in soup.find_all(tag_name):
                    tag.decompose()
            soup = self._remove_gmail_quotes(soup)
            for element in soup.find_all(attrs={"class": re.compile(r"reply|forward|quote|original", re.I)}):
                element.decompose()
            for hr in soup.find_all("hr"):
                hr.decompose()
            return str(soup)
        except Exception as e:
            raise PreprocessingError(f"Failed to clean HTML: {e}") from e

    def _html_to_text(self, html_content: str) -> str:
        """Convert HTML to plain text with formatting preserved."""
        try:
            soup = BeautifulSoup(html_content, "lxml")
            for tag in soup.find_all(["p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6"]):
                if tag.name == "br":
                    tag.replace_with("\n")
                else:
                    tag.append("\n")
            return soup.get_text(separator=" ", strip=True)
        except Exception as e:
            raise PreprocessingError(f"Failed to convert HTML to text: {e}") from e

    def _normalize_text(self, text: str) -> str:
        """Normalize text: unicode, whitespace, line endings, null bytes."""
        text = unicodedata.normalize("NFC", text)
        text = text.replace("\x00", "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = text.split("\n")
        normalized_lines: list[str] = []
        prev_empty = False
        for line in lines:
            normalized_line = " ".join(line.split())
            if normalized_line:
                normalized_lines.append(normalized_line)
                prev_empty = False
            elif not prev_empty:
                normalized_lines.append("")
                prev_empty = True
        return "\n".join(normalized_lines).strip()

    def _remove_disclaimers(self, text: str) -> str:
        """Remove common legal disclaimers from email text."""
        original = text
        for pattern in DISCLAIMER_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
        if text != original:
            logger.debug("Removed disclaimer content")
        return text.strip()
