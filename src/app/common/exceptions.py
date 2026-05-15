"""Custom exceptions for the email extraction platform."""


class EmailExtractionError(Exception):
    """Base exception for email extraction errors."""


class PreprocessingError(EmailExtractionError):
    """Raised when email preprocessing fails."""


class InvalidEmailError(EmailExtractionError):
    """Raised when the provided email is invalid or malformed."""


class EmptyContentError(EmailExtractionError):
    """Raised when email content is empty after preprocessing."""


class ProviderError(EmailExtractionError):
    """Raised when LLM provider invocation fails."""


class RetriableProviderError(ProviderError):
    """Raised for transient provider failures that are safe to retry (rate limits, timeouts, 5xx)."""


class ProviderNotConfiguredError(ProviderError):
    """Raised when the requested LLM provider is not configured."""


class ConfidenceScoreError(EmailExtractionError):
    """Raised when confidence scoring fails."""


class AgentExecutionError(EmailExtractionError):
    """Raised when LLM agent execution fails."""
