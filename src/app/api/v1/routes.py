"""FastAPI Application v1 routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.config.settings import get_settings
from app.common.logging_config import logger
from app.models.schemas import (
    ExtractionRequest,
    ExtractionResponse,
    HealthResponse,
    HealthStatus,
    ProvidersResponse,
)
from app.providers.factory import list_available_providers
from app.services.extraction_service import ExtractionService
from app.services.history_service import HistoryEntry, clear_history, load_history

router = APIRouter()

_EXTRACTION_SERVICE: ExtractionService = ExtractionService()


def get_extraction_service() -> ExtractionService:
    """Get the extraction service instance."""
    return _EXTRACTION_SERVICE


@router.get(
    path="/health",
    response_model=HealthResponse,
    tags=["Application"],
    summary="Health check",
    description="Check service health status and version information.",
)
async def health_check() -> HealthResponse:
    """Return service health status and version."""
    settings = get_settings()
    return HealthResponse(status=HealthStatus.HEALTHY, version=settings.app_version)


@router.post(
    path="/extract",
    response_model=ExtractionResponse,
    tags=["Extraction"],
    summary="Extract latest message from email",
    description=(
        "Extract the latest message from an email thread. "
        "Supports three modes: `non_llm` (deterministic), `llm` (agentic), and "
        "`auto` (non_llm first, falls back to llm if confidence is below threshold)."
    ),
)
async def extract_email(
    request: ExtractionRequest,
    service: Annotated[ExtractionService, Depends(get_extraction_service)],
) -> ExtractionResponse:
    """Extract the latest message from an email thread."""
    logger.info("API request: POST /extract")
    response = service.extract(request)
    if not response.success:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=response.error or "Extraction failed",
        )
    return response


@router.post(
    path="/extract/stream",
    tags=["Extraction"],
    summary="Stream extraction results as Server-Sent Events",
    description=(
        "Stream the extraction process using Server-Sent Events (SSE). "
        "Emits events for each step of the extraction, including intermediate tokens "
        "from the LLM when in `llm` mode. Useful for real-time feedback in the UI."
        "Events:\n"
        '- `{"type": "start", "agent": "<name>"}`: Indicates the start of an agent\'s execution.\n'
        '- `{"type": "token", "text": "<chunk>"}`: A chunk of text output from the LLM (only in `llm` mode).\n'
        '- `{"type": "agent_done", "agent": "<name>"}`: Indicates an agent has completed its execution.\n'
        '- `{"type": "done", "result": {...}}`: The final extraction result after all processing is complete.\n'
        '- `{"type": "error", "message": "<msg>"}`: An error occurred during processing, with a message describing the '
        "issue."
    ),
)
def extract_email_stream(
    request: ExtractionRequest,
    service: Annotated[ExtractionService, Depends(get_extraction_service)],
) -> StreamingResponse:
    """Stream extraction results as Server-Sent Events."""
    logger.info("API request: POST /extract/stream")
    return StreamingResponse(service.stream(request), media_type="text/event-stream")


@router.get(
    path="/history",
    response_model=list[HistoryEntry],
    tags=["History"],
    summary="List extraction history",
    description="Return a list of past extraction attempts, including input metadata and results. ",
)
def get_history() -> list[HistoryEntry]:
    """Return extraction history (latest first, up to 100 entries)."""
    return load_history()


@router.delete(
    path="/history",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["History"],
    summary="Clear extraction history",
    description="Delete all extraction history entries. This action cannot be undone.",
)
def delete_history() -> None:
    """Clear all extraction history."""
    clear_history()


@router.get(
    path="/providers",
    response_model=ProvidersResponse,
    tags=["Providers"],
    summary="List available LLM providers",
    description="Returns a list of LLM providers and their configuration status.",
)
async def list_providers() -> ProvidersResponse:
    """List available LLM providers and their status."""
    providers_status = list_available_providers()
    default = next((provider.name for provider in providers_status if provider.available), "none")
    return ProvidersResponse(providers=providers_status, default_provider=default)
