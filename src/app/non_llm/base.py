"""Base interfaces for non-LLM text cleaning components."""

from abc import ABC, abstractmethod
from typing import ClassVar


class TextCleaner(ABC):
    """Abstract base for text cleaning components used in the non-LLM pipeline."""

    step_name: ClassVar[str] = ""

    @abstractmethod
    def clean(self, text: str) -> tuple[str, bool]:
        """Clean the text, returning (cleaned_text, was_modified)."""
