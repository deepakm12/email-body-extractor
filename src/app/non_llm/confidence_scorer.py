"""Confidence scoring for non-LLM extraction results."""

import re
from dataclasses import dataclass

from app.common.logging_config import logger

_NOISE_INDICATORS = [
    r"<[^>]+>",
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    r"https?://\S+",
    r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    r"^\s*>",
    r"Forwarded by",
    r"Original Message",
    r"^\s*Sent from my",
    r"\b(CONFIDENTIAL|DISCLAIMER|NOTICE)\b",
]

_QUALITY_INDICATORS = [
    r"[.!?]\s+[A-Z]",
    r"\n\n",
    r"\b(the|a|an|is|are|was|were|have|has|had|do|does|did|will|would|could|should)\b",
    r"[?!]",
    r"\b(Hi|Hello|Hey|Dear|Good morning|Good afternoon|Good evening)\b",
    r"\b(Thanks|Thank you|Best|Regards|Cheers|Sincerely)\b",
]

_NOISE_PENALTY = 0.05
_QUALITY_REWARD = 0.03
_MIN_CONTENT_LENGTH = 10
_MAX_CONTENT_LENGTH = 10000


@dataclass
class ConfidenceResult:
    """Result of confidence scoring."""

    score: float
    noise_found: list[str]
    quality_found: list[str]
    details: dict[str, object]
    is_reliable: bool = False


class ConfidenceScorer:
    """Calculates confidence scores for extracted text."""

    _threshold: float

    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def score(self, text: str) -> ConfidenceResult:
        """Score the extracted text and determine if it's reliable."""
        if not text or not text.strip():
            return ConfidenceResult(
                score=0.0,
                noise_found=[],
                quality_found=[],
                is_reliable=False,
                details={"error": "Empty result"},
            )
        text = text.strip()
        length_score = self._content_length_score(text)
        noise_score, noise_found = self._noise_score(text)
        quality_score, quality_found = self._quality_score(text)
        readability_score = self._readability_score(text)
        completeness_score = self._completeness_score(text)
        final_score = (
            length_score * 0.20
            + noise_score * 0.30
            + quality_score * 0.20
            + readability_score * 0.15
            + completeness_score * 0.15
        )
        result = ConfidenceResult(
            score=round(min(1.0, max(0.0, final_score)), 3),
            details={
                "length_score": round(length_score, 3),
                "noise_score": round(noise_score, 3),
                "quality_score": round(quality_score, 3),
                "readability_score": round(readability_score, 3),
                "completeness_score": round(completeness_score, 3),
            },
            noise_found=noise_found,
            quality_found=quality_found,
            is_reliable=final_score >= self._threshold,
        )
        logger.info(
            "Confidence scoring complete",
            extra={"score": result.score, "is_reliable": result.is_reliable, "threshold": self._threshold},
        )
        return result

    def _content_length_score(self, text: str) -> float:
        """Calculate a score based on the length of the content, penalizing very short or excessively long text."""
        length = len(text.strip())
        if length < _MIN_CONTENT_LENGTH:
            return 0.1
        if length < 50:
            return 0.6
        if length < 200:
            return 0.9
        if length > _MAX_CONTENT_LENGTH:
            return max(0.5, 1.0 - (length - _MAX_CONTENT_LENGTH) / 50000)
        return 1.0

    def _noise_score(self, text: str) -> tuple[float, list[str]]:
        """Penalize for indicators of noisy or irrelevant text."""
        noise_found: list[str] = []
        penalty = 0.0
        for pattern in _NOISE_INDICATORS:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                noise_found.append(f"{pattern}: {len(matches)} matches")
                penalty += min(len(matches) * _NOISE_PENALTY, 0.2)
        return max(0.0, 1.0 - penalty), noise_found

    def _quality_score(self, text: str) -> tuple[float, list[str]]:
        """Reward for indicators of good quality text."""
        quality_found: list[str] = []
        reward = 0.0
        for pattern in _QUALITY_INDICATORS:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                quality_found.append(f"{pattern}: {len(matches)} matches")
                reward += min(len(matches) * _QUALITY_REWARD, 0.1)
        return min(1.0, 0.5 + reward), quality_found

    def _readability_score(self, text: str) -> float:
        """Rough heuristic for readability based on special character ratio and word density."""
        special_char_ratio = len(re.findall(r"[^\w\s.,;:!?'\"-]", text)) / max(len(text), 1)
        if special_char_ratio > 0.1:
            return 0.6
        words = len(text.split())
        chars = len(text)
        if chars == 0:
            return 0.0
        ratio = words / chars
        if ratio < 0.1:
            return 0.7
        return 1.0

    def _completeness_score(self, text: str) -> float:
        """Check if the text seems complete based on punctuation and structure."""
        stripped = text.strip()
        if stripped and not re.search(r"[.!?]$", stripped[-1:]):
            last_chunk = stripped[-50:]
            if not re.search(r"[.!?]", last_chunk):
                return 0.7
        first_char = stripped[0] if stripped else ""
        if first_char in ",.;:!?":
            return 0.8
        return 1.0
