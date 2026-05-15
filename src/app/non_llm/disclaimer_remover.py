"""Disclaimer and legal notice removal."""

import re

from app.non_llm.base import TextCleaner
from app.common.logging_config import logger

# Disclaimer start markers - must appear as standalone headers
# Use word boundaries and require they appear at start of line or after whitespace
_DISCLAIMER_START_PATTERNS = [
    r"(?:^|\n)\s*CONFIDENTIALITY NOTICE\s*[:\n]",
    r"(?:^|\n)\s*DISCLAIMER\s*[:\n]",
    r"(?:^|\n)\s*LEGAL NOTICE\s*[:\n]",
    r"(?:^|\n)\s*PRIVACY NOTICE\s*[:\n]",
    r"(?:^|\n)\s*IMPORTANT NOTICE\s*[:\n]",
    r"(?:^|\n)\s*CAUTION\s*[:\n]",
    r"(?:^|\n)\s*WARNING\s*[:\n]",
    r"(?:^|\n)\s*ATTENTION\s*[:\n]",
]

# Disclaimer content patterns
_DISCLAIMER_PATTERNS = [
    # Comprehensive confidentiality
    r"CONFIDENTIALITY NOTICE:?\s*\n.*?\n\s*\n",
    r"DISCLAIMER:?\s*\n.*?\n\s*\n",
    # Common legal phrases
    (
        r"This\s+(?:e-?mail|message|communication|transmission)\s+"
        r"(?:is\s+)?(?:intended\s+only\s+for|confidential|strictly\s+confidential)"
        r".*?(?:\n\s*\n|\Z)"
    ),
    r"If\s+you\s+have\s+received\s+this\s+(?:e-?mail|message).*?(?:\n\s*\n|\Z)",
    r"The\s+information\s+contained\s+in\s+this\s+(?:e-?mail|message).*?(?:\n\s*\n|\Z)",
    r"This\s+(?:e-?mail|message)\s+and\s+any\s+attachments\s+are\s+confidential.*?(?:\n\s*\n|\Z)",
    # Virus warning
    r"Although\s+we\s+have\s+taken\s+reasonable\s+precautions.*?virus.*?(?:\n\s*\n|\Z)",
    r"No\s+liability\s+is\s+accepted\s+for\s+any\s+opinions.*?(?:\n\s*\n|\Z)",
    # Environmental
    r"Please\s+consider\s+the\s+environment\s+before\s+printing.*?(?:\n|\Z)",
    # GDPR/EU
    r"(?:GDPR|EU General Data Protection Regulation).*?(?:\n\s*\n|\Z)",
    r"(?:data protection|personal data).*?(?:rights|regulation).*?(?:\n\s*\n|\Z)",
]

# Separator lines that often precede disclaimers
_DISCLAIMER_SEPARATORS = [
    r"-{10,}",
    r"_{10,}",
    r"\*{10,}",
    r"={10,}",
]


def _find_disclaimer_start(text: str) -> int:
    """Find the starting position of disclaimer text."""
    earliest = -1
    for pattern in _DISCLAIMER_START_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if earliest == -1 or match.start() < earliest:
                earliest = match.start()
    return earliest


def _remove_with_patterns(text: str) -> str:
    """Remove disclaimers using regex patterns."""
    for pattern in _DISCLAIMER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _remove_by_position(text: str) -> str:
    """Remove disclaimer by finding its start position."""
    start = _find_disclaimer_start(text)
    if start == -1:
        return text
    # Check if there's a separator line just before the disclaimer
    for sep in _DISCLAIMER_SEPARATORS:
        # Look for separator within 100 chars before disclaimer
        search_start = max(0, start - 100)
        sep_match = re.search(sep, text[search_start:start])
        if sep_match:
            start = search_start + sep_match.start()
            break
    result = text[:start].strip()
    logger.debug(f"Removed disclaimer by position: removed {len(text) - len(result)} chars")
    return result


def _remove_disclaimer(text: str) -> tuple[str, bool]:
    """Remove legal disclaimers from email text."""
    original = text
    # Pass 1: Remove by patterns
    text = _remove_with_patterns(text)
    # Pass 2: Remove by position (catches trailing disclaimers)
    text = _remove_by_position(text)
    was_modified = len(text) < len(original)
    logger.info(
        "Disclaimer removal complete",
        extra={"original_length": len(original), "cleaned_length": len(text), "was_modified": was_modified},
    )
    return text, was_modified


class DisclaimerRemover(TextCleaner):
    """Removes legal disclaimers from email text."""

    step_name = "disclaimer_removal"

    def clean(self, text: str) -> tuple[str, bool]:
        return _remove_disclaimer(text)
