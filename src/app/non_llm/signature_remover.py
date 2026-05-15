"""Signature removal using Talon with regex fallback."""

import re

import talon
from talon.signature.bruteforce import extract_signature as _talon_extract_signature

from app.non_llm.base import TextCleaner
from app.common.logging_config import logger

talon.init()

# Common signature patterns (fallback when Talon is unavailable)
_SIGNATURE_PATTERNS = [
    # Name and title
    r"^\s*--\s*\n.*?(?:\n|$)",
    r"^\s*—\s*\n.*?(?:\n|$)",
    r"^\s*__\s*\n.*?(?:\n|$)",
    # Common signature keywords
    r"(?:^|\n)\s*(?:Best|Regards|Cheers|Thanks|Thank you|Sincerely|Yours truly),?\s*\n+\s*\S+",
    r"(?:^|\n)\s*(?:Warm regards|Kind regards|Best wishes),?\s*\n+\s*\S+",
    # Phone/email patterns in signatures
    r"(?:^|\n)\s*(?:Tel|Phone|Mobile|Cell):\s*[\d\s\-+()]+\s*\n",
    r"(?:^|\n)\s*(?:Email|E-mail):\s*\S+@\S+\s*\n",
    # Social media / web
    r"(?:^|\n)\s*(?:LinkedIn|Twitter|Skype):\s*\S+\s*\n",
    r"(?:^|\n)\s*www\.\S+\s*\n",
    # Legal entity markers
    r"(?:^|\n)\s*(?:Ltd\.|LLC|Inc\.|Corp\.|GmbH)\s*\n",
]

# Lines that commonly appear in signatures
_SIGNATURE_LINE_PATTERNS = [
    r"^\s*Sent from my \w+",
    r"^\s*Sent from my iPhone",
    r"^\s*Sent from my Android",
    r"^\s*Sent from my BlackBerry",
    r"^\s*Sent from my Windows Phone",
    r"^\s*Sent from my iPad",
    r"^\s*Sent from my mobile device",
    r"^\s*Sent from Mail for Windows",
    r"^\s*Sent from Outlook",
    r"^\s*Sent from Thunderbird",
    r"^\s*Get Outlook for \w+",
    r"^\s*Download the Outlook app",
]


def _remove_with_talon(text: str) -> tuple[str, bool]:
    """Remove signature using Talon library."""
    try:
        body, signature = _talon_extract_signature(text)
        if signature and len(signature.strip()) > 0:
            logger.debug(
                "Talon removed signature",
                extra={"signature_length": len(signature), "signature_preview": signature[:100]},
            )
            return body.strip(), True
        return text, False
    except Exception as e:
        logger.warning(f"Talon extraction failed: {e}")
        return text, False


def _remove_with_regex(text: str) -> tuple[str, bool]:
    """Remove signature using regex patterns."""
    original = text
    lines = text.split("\n")
    signature_start_index = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^\s*(--|—|__)\s*$", stripped):
            signature_start_index = i
            break
        # closing phrases followed by a name
        if re.match(r"^(Best|Regards|Cheers|Thanks|Thank you|Sincerely|Yours truly),?\s*$", stripped, re.I):
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and len(next_line) < 60 and not re.match(r"^[a-z]", next_line):
                    signature_start_index = i
                    break
    if signature_start_index == len(lines):
        for i, line in enumerate(lines):
            for pattern in _SIGNATURE_LINE_PATTERNS:
                if re.match(pattern, line, re.I):
                    signature_start_index = i
                    break
            if signature_start_index < len(lines):
                break
    cleaned_lines = lines[:signature_start_index]
    partial_text = "\n".join(cleaned_lines)
    for pattern in _SIGNATURE_PATTERNS:
        match = re.search(pattern, partial_text, re.MULTILINE | re.IGNORECASE)
        if match:
            # Truncate at the first matching signature block
            partial_text = partial_text[: match.start()].rstrip()
            logger.debug("_SIGNATURE_PATTERNS matched: %s", pattern)
            break  # one pass is enough; patterns are ordered from most to least specific
    cleaned_lines = partial_text.split("\n")
    # Remove any stray "Sent from…" lines that slipped through inline
    final_lines = [line for line in cleaned_lines if not any(re.match(p, line, re.I) for p in _SIGNATURE_LINE_PATTERNS)]
    result = "\n".join(final_lines).strip()
    if len(result) < len(original):
        logger.debug("Regex signature removal removed %d chars", len(original) - len(result))
    return result, len(result) < len(original)


def _remove_signature(text: str) -> tuple[str, bool]:
    """Remove email signature."""
    original = text
    text, was_modified = _remove_with_talon(text)
    if was_modified:
        return text, True
    # Fallback to regex
    text, was_modified = _remove_with_regex(text)
    # Clean up any extra whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    logger.info(
        "Signature removal complete",
        extra={"original_length": len(original), "cleaned_length": len(text), "was_modified": was_modified},
    )
    return text, was_modified


class SignatureRemover(TextCleaner):
    """Removes email signatures."""

    step_name = "signature_removal"

    def clean(self, text: str) -> tuple[str, bool]:
        return _remove_signature(text)
