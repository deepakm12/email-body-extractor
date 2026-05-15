"""Extraction router that orchestrates between non-LLM and LLM flows."""

from dataclasses import dataclass
from enum import Enum

from app.config.settings import AppSettings, get_settings
from app.llm_flow.workflow import LLMFlowResult, run_llm_flow
from app.models.schemas import ExtractionMode as RequestMode
from app.non_llm.pipeline import run_pipeline
from app.providers.factory import get_provider
from app.common.logging_config import logger


class FlowUsed(str, Enum):
    """Flow used for extraction, for observability and debugging purposes."""

    LLM = "llm"
    NON_LLM = "non_llm"
    AUTO_LLM = "llm (auto mode)"
    AUTO_NON_LLM = "non_llm (auto mode)"
    AUTO_NON_LLM_FAILED = "non_llm (auto mode, llm failed)"


@dataclass
class RouterResult:
    """Typed result returned by every ExtractionRouter path."""

    success: bool
    flow_used: FlowUsed
    confidence: float
    latest_message: str
    metadata: dict[str, object]
    error: str | None = None
    agent_trace: list[dict[str, object]] | None = None


class ExtractionRouter:
    """Router for email extraction pipelines."""

    _settings: AppSettings

    def __init__(self) -> None:
        self._settings = get_settings()

    def extract(self, content: str, mode: RequestMode, provider_name: str | None = None) -> RouterResult:
        """Route extraction request to appropriate pipeline."""
        logger.info("Routing extraction request", extra={"mode": mode.value, "provider": provider_name})
        match mode:
            case RequestMode.NON_LLM:
                return self._run_non_llm(content=content)
            case RequestMode.LLM:
                return self._run_llm(content=content, provider_name=provider_name)
            case RequestMode.AUTO:
                return self._run_auto(content=content, provider_name=provider_name)
            case _:
                raise ValueError(f"Unsupported extraction mode: {mode}")

    def _run_non_llm(self, content: str) -> RouterResult:
        """Run the non-LLM deterministic pipeline."""
        logger.info("Running non-LLM pipeline")
        result = run_pipeline(text=content, threshold=self._settings.confidence_threshold)
        return RouterResult(
            latest_message=result.text,
            confidence=result.confidence.score,
            flow_used=FlowUsed.NON_LLM,
            metadata=result.metadata,
            success=True,
        )

    def _run_llm(self, content: str, provider_name: str | None = None) -> RouterResult:
        """Run the LLM agent flow."""
        logger.info("Running LLM pipeline")
        try:
            provider = get_provider(provider_name=provider_name)
            result: LLMFlowResult = run_llm_flow(content, provider)
            return RouterResult(
                latest_message=result.text,
                confidence=result.confidence,
                flow_used=FlowUsed.LLM,
                metadata={"iterations": result.iterations, "llm_provider": provider.get_metadata()},
                agent_trace=result.agent_trace,
                success=result.success,
            )
        except Exception as e:
            logger.error("LLM pipeline failed: %s", e)
            return RouterResult(
                latest_message="",
                confidence=0.0,
                flow_used=FlowUsed.LLM,
                metadata={"error": str(e)},
                success=False,
                error=str(e),
            )

    def _run_auto(self, content: str, provider_name: str | None = None) -> RouterResult:
        """Run auto mode: non-LLM first, LLM fallback if needed."""
        logger.info("Running AUTO mode: non-LLM first")
        non_llm_result = run_pipeline(text=content, threshold=self._settings.confidence_threshold)
        logger.info(
            "Non-LLM result",
            extra={"confidence": non_llm_result.confidence.score, "is_reliable": non_llm_result.confidence.is_reliable},
        )
        if non_llm_result.confidence.is_reliable:
            return RouterResult(
                latest_message=non_llm_result.text,
                confidence=non_llm_result.confidence.score,
                flow_used=FlowUsed.AUTO_NON_LLM,
                metadata=non_llm_result.metadata,
                success=True,
            )
        logger.info(
            f"Non-LLM confidence ({non_llm_result.confidence.score}) below threshold "
            f"({self._settings.confidence_threshold}), falling back to LLM"
        )
        llm_result = self._run_llm(content, provider_name)
        if not llm_result.success:
            logger.warning("LLM fallback failed, returning non-LLM result")
            return RouterResult(
                latest_message=non_llm_result.text,
                confidence=non_llm_result.confidence.score,
                flow_used=FlowUsed.AUTO_NON_LLM_FAILED,
                metadata={**non_llm_result.metadata, "llm_fallback_error": llm_result.error},
                success=True,
            )
        return RouterResult(
            latest_message=llm_result.latest_message,
            confidence=llm_result.confidence,
            flow_used=FlowUsed.AUTO_LLM,
            metadata={
                **llm_result.metadata,
                "non_llm_fallback": {
                    "non_llm_confidence": non_llm_result.confidence.score,
                    "non_llm_text": non_llm_result.text,
                },
            },
            agent_trace=llm_result.agent_trace,
            success=True,
        )
