"""Extraction service that orchestrates preprocessing and routing."""

import json
from collections.abc import Generator

from app.config.settings import AppSettings, get_settings
from app.llm_flow.agents import EXTRACTION_SYSTEM, cleanup_agent, confidence_agent
from app.common.logging_config import logger
from app.models.schemas import ExtractionMode, ExtractionRequest, ExtractionResponse, ExtractionResult
from app.non_llm.pipeline import run_pipeline
from app.providers.base import BaseLLMProvider
from app.providers.factory import get_provider
from app.router.extraction_router import ExtractionRouter, RouterResult
from app.services.history_service import save_entry
from app.common.exceptions import EmailExtractionError
from app.common.preprocessing import EmailPreprocessor


def _sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


class ExtractionService:
    """Service layer for email extraction."""

    _settings: AppSettings
    _router: ExtractionRouter
    _email_preprocessor: EmailPreprocessor

    def __init__(self) -> None:
        self._settings = get_settings()
        self._router = ExtractionRouter()
        self._email_preprocessor = EmailPreprocessor()

    def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        """Process an extraction request."""
        logger.info(
            "Processing extraction request",
            extra={
                "mode": request.mode.value,
                "content_length": len(request.content),
                "is_html": "<html" in request.content.lower() or "<body" in request.content.lower(),
            },
        )
        try:
            text_content = self._preprocess(request)
            if text_content is None:
                return ExtractionResponse(success=False, error="Empty content after preprocessing", data=None)
            result = self._router.extract(content=text_content, mode=request.mode, provider_name=request.provider)
            if not result.success:
                return ExtractionResponse(success=False, error=result.error or "Extraction failed", data=None)
            response = ExtractionResponse(success=True, data=self._build_result(result), agent_trace=result.agent_trace)
            self._try_save(result, request)
            return response
        except EmailExtractionError as e:
            logger.error(f"Extraction error: {e}")
            return ExtractionResponse(success=False, error=str(e), data=None)
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return ExtractionResponse(success=False, error=f"Internal error: {e}", data=None)

    def stream(self, request: ExtractionRequest) -> Generator[str, None, None]:
        """Yield SSE events for a streaming extraction request."""
        text_content = self._preprocess(request)
        if text_content is None:
            yield _sse({"type": "error", "message": "Empty content after preprocessing"})
            return
        if request.mode == ExtractionMode.NON_LLM:
            yield from self._stream_non_llm(text_content)
            return
        try:
            provider = get_provider(provider_name=request.provider)
        except Exception as e:
            yield _sse({"type": "error", "message": str(e)})
            return
        yield from self._stream_llm(provider, text_content)

    def _preprocess(self, request: ExtractionRequest) -> str | None:
        """Return preprocessed plain text, or None if empty after preprocessing."""
        processed = self._email_preprocessor.process(request.content, is_eml=request.is_eml)
        text_content = processed.text
        if not text_content.strip():
            return None
        logger.debug(f"Preprocessing complete: {len(request.content)} -> {len(text_content)} chars")
        return text_content

    def _build_result(self, result: RouterResult) -> ExtractionResult:
        return ExtractionResult(
            metadata=result.metadata,
            flow_used=result.flow_used,
            confidence=result.confidence,
            latest_message=result.latest_message,
        )

    def _try_save(self, result: RouterResult, request: ExtractionRequest) -> None:
        try:
            save_entry(result=result, request=request)
        except Exception:
            pass

    def _stream_non_llm(self, text_content: str) -> Generator[str, None, None]:
        yield _sse(payload={"type": "start", "agent": "non_llm"})
        result = run_pipeline(text_content, threshold=self._settings.confidence_threshold)
        yield _sse(payload={"type": "token", "text": result.text})
        yield _sse(
            payload={
                "type": "done",
                "result": {
                    "latest_message": result.text,
                    "confidence": result.confidence.score,
                    "flow_used": "non_llm",
                },
            }
        )

    def _stream_llm(self, provider: BaseLLMProvider, text_content: str) -> Generator[str, None, None]:
        yield _sse(payload={"type": "start", "agent": "extraction"})
        prompt = (
            "Extract the latest email message from this content."
            f" Return ONLY the message, no explanations:\n\n{text_content}"
        )
        extracted_text = ""
        try:
            for token in provider.stream(prompt=prompt, system_message=EXTRACTION_SYSTEM):
                extracted_text += token
                yield _sse(payload={"type": "token", "text": token})
        except Exception as e:
            yield _sse(payload={"type": "error", "message": str(e)})
            return
        yield _sse(payload={"type": "agent_done", "agent": "extraction"})
        yield _sse(payload={"type": "start", "agent": "cleanup"})
        cleanup_result = cleanup_agent(provider, extracted_text)
        yield _sse(payload={"type": "agent_done", "agent": "cleanup"})
        yield _sse(payload={"type": "start", "agent": "confidence"})
        confidence_result = confidence_agent(provider, cleanup_result.output)
        score_raw = confidence_result.metadata.get("confidence_score", 0.5)
        confidence_score = float(score_raw) if isinstance(score_raw, (int, float)) else 0.5
        yield _sse(payload={"type": "agent_done", "agent": "confidence"})
        yield _sse(
            payload={
                "type": "done",
                "result": {
                    "latest_message": cleanup_result.output,
                    "confidence": confidence_score,
                    "flow_used": "llm_stream",
                },
            }
        )
