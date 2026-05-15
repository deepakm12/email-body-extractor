"""Pydantic models for API requests and responses."""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ExtractionMode(str, Enum):
    """Supported extraction modes."""

    NON_LLM = "non_llm"
    LLM = "llm"
    AUTO = "auto"


class HealthStatus(str, Enum):
    """Health status values."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class ExtractionRequest(BaseModel):
    """Request model for the extraction endpoint."""

    content: str = Field(..., description="Email content (HTML, plain text, or .eml string)", min_length=1)
    mode: ExtractionMode = Field(
        default=ExtractionMode.AUTO,
        description="Extraction mode: non_llm, llm, or auto",
    )
    provider: str | None = Field(
        default=None,
        description="LLM provider override (openai, azure_openai, anthropic, gemini)",
    )
    is_eml: bool = Field(
        default=False,
        description="Whether the content is an .eml file (decoded as string)",
    )

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        """Validate that content is not empty or whitespace only."""
        if not v or not v.strip():
            raise ValueError("Content cannot be empty or whitespace only")
        return v


class ExtractionResult(BaseModel):
    """Result model for email extraction."""

    latest_message: str = Field(..., description="Extracted latest email message")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    flow_used: str = Field(..., description="Pipeline used: non_llm or llm")
    metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Additional metadata about extraction",
    )


class ExtractionResponse(BaseModel):
    """Response model for the extraction endpoint."""

    success: bool = Field(..., description="Whether extraction was successful")
    data: ExtractionResult | None = None
    error: str | None = None
    agent_trace: list[dict[str, object]] | None = Field(
        default=None,
        description="Trace of LLM agent execution steps",
    )


class HealthResponse(BaseModel):
    """Health check response model."""

    version: str
    status: HealthStatus


class ProviderInfo(BaseModel):
    """Provider information model."""

    name: str
    available: bool
    configured: bool


class ProvidersResponse(BaseModel):
    """Response model for available providers."""

    providers: list[ProviderInfo]
    default_provider: str
