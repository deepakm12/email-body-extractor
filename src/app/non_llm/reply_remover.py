"""Reply removal using email-reply-parser and regex heuristics."""

import re

from email_reply_parser import EmailReplyParser

from app.non_llm.base import TextCleaner
from app.common.logging_config import logger

# Common reply patterns across email clients
_REPLY_PATTERNS = [
    # Gmail web
    r"On\s+.*?\d{1,2},?\s+\d{4}.*?(?:at\s+.*?)?wrote:",
    # Outlook
    r"From:\s*[^\n]+\n\s*Sent:\s*[^\n]+\n\s*To:\s*[^\n]+\n\s*Subject:\s*[^\n]+",
    # Apple Mail
    r">?\s*On\s+.*?wrote:",
    # Thunderbird
    r"-{3,}\s*Original Message\s*-{3,}",
    # Yahoo Mail
    r"_{3,}\nFrom:.*?\nDate:.*?\nTo:.*?\nSubject:.*?\n",
    # Generic forwarded
    r"\n\s*>\s+",  # Blockquoted lines
    r"Begin forwarded message:",
    r"Forwarded by.*?on.*?\n",
    # Separator lines
    r"\n-{5,}\n",
    r"\n_{5,}\n",
]

# Reply headers that indicate the start of a quoted reply
_REPLY_HEADER_PATTERNS = [
    r"^\s*>\s*From:\s*",
    r"^\s*>\s*Sent:\s*",
    r"^\s*>\s*To:\s*",
    r"^\s*>\s*Subject:\s*",
    r"^\s*>\s*Date:\s*",
    r"^\s*>\s*Cc:\s*",
]


def _remove_blockquoted_lines(text: str) -> str:
    """Remove blockquoted lines (lines starting with >)."""
    lines = text.split("\n")
    cleaned_lines: list[str] = []
    in_quote = False
    for line in lines:
        # Check if this is a blockquoted line
        if re.match(r"^\s*>", line):
            in_quote = True
            continue
        # Check if this is a reply header line
        is_header = False
        for pattern in _REPLY_HEADER_PATTERNS:
            if re.match(pattern, line, re.I):
                is_header = True
                in_quote = True
                break
        if is_header:
            continue
        # If we were in a quote block and hit an empty line, stay cautious
        if in_quote and line.strip() == "":
            continue
        in_quote = False
        cleaned_lines.append(line)
    result = "\n".join(cleaned_lines)
    if len(result) < len(text):
        logger.debug(f"Removed blockquoted lines: {len(text) - len(result)} chars")
    return result


def _remove_reply_headers(text: str) -> str:
    """Remove reply/forward headers using regex patterns."""
    original = text
    for pattern in _REPLY_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    # Remove consecutive blank lines created by removal
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) < len(original):
        logger.debug(f"Removed reply headers: {len(original) - len(text)} chars")
    return text.strip()


def _parse_with_email_reply_parser(text: str) -> str:
    """Use email-reply-parser library for reply extraction."""
    try:
        parsed = EmailReplyParser.read(text)
        fragments = parsed.fragments
        # Collect non-hidden (non-quoted) fragments
        latest_parts: list[str] = []
        for fragment in fragments:
            if not fragment.hidden:
                content = fragment.content.strip()
                if content:
                    latest_parts.append(content)
        result = "\n\n".join(latest_parts)
        if not result and text.strip():
            # Check if the original text is a short standalone response
            first_line = text.strip().split("\n")[0].strip()
            if len(first_line) <= 20 and "On " not in first_line:
                return first_line
        logger.debug(
            "email-reply-parser results",
            extra={
                "total_fragments": len(fragments),
                "visible_fragments": len(latest_parts),
                "result_length": len(result),
            },
        )
        return result
    except Exception as e:
        logger.warning(f"email-reply-parser failed: {e}, falling back")
        return text


def _remove_replies(text: str) -> tuple[str, bool]:
    """Remove quoted replies from email text."""
    original = text
    # Pass 1: Use email-reply-parser
    text = _parse_with_email_reply_parser(text)
    # Pass 2: Remove blockquoted lines (catches what library misses)
    text = _remove_blockquoted_lines(text)
    # Pass 3: Remove reply headers with regex
    text = _remove_reply_headers(text)
    # Final cleanup
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    was_modified = len(text) < len(original) or text != original
    logger.info(
        "Reply removal complete",
        extra={"original_length": len(original), "cleaned_length": len(text), "was_modified": was_modified},
    )
    return text, was_modified


class ReplyRemover(TextCleaner):
    """Removes quoted replies from email text."""

    step_name = "reply_removal"

    def clean(self, text: str) -> tuple[str, bool]:
        return _remove_replies(text)
