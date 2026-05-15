"""Non-LLM extraction pipeline orchestrator."""

from dataclasses import dataclass

from app.non_llm.base import TextCleaner
from app.non_llm.confidence_scorer import ConfidenceResult, ConfidenceScorer
from app.non_llm.disclaimer_remover import DisclaimerRemover
from app.non_llm.reply_remover import ReplyRemover
from app.non_llm.signature_remover import SignatureRemover
from app.common.logging_config import logger


@dataclass
class NonLLMResult:
    """Result of non-LLM extraction pipeline."""

    text: str
    confidence: ConfidenceResult
    steps_executed: list[str]
    metadata: dict[str, object]


class NonLLMPipeline:
    """Orchestrates pluggable text cleaning steps for non-LLM extraction."""

    _threshold: float
    _scorer: ConfidenceScorer
    _steps: list[TextCleaner]

    def __init__(self, threshold: float, steps: list[TextCleaner] | None = None) -> None:
        self._threshold = threshold
        self._scorer = ConfidenceScorer(threshold=threshold)
        self._steps: list[TextCleaner] = steps or [ReplyRemover(), SignatureRemover(), DisclaimerRemover()]

    def run(self, text: str) -> NonLLMResult:
        """Run all cleaning steps and return scored result."""
        original_length = len(text)
        steps_executed: list[str] = []
        metadata: dict[str, object] = {"original_length": original_length}
        logger.info(
            "Starting non-LLM extraction pipeline",
            extra={"threshold": self._threshold, "text_length": original_length},
        )
        for step in self._steps:
            text, was_modified = step.clean(text)
            if was_modified:
                steps_executed.append(step.step_name)
            metadata[f"after_{step.step_name}_length"] = len(text)
        confidence = self._scorer.score(text)
        steps_executed.append("confidence_scoring")
        metadata.update(
            {
                "confidence_score": confidence.score,
                "is_reliable": confidence.is_reliable,
                "confidence_details": confidence.details,
                "noise_found": confidence.noise_found,
                "quality_found": confidence.quality_found,
                "reduction_ratio": (
                    round((original_length - len(text)) / original_length, 3) if original_length > 0 else 0
                ),
            }
        )
        logger.info(
            "Non-LLM pipeline complete",
            extra={
                "steps": steps_executed,
                "final_length": len(text),
                "confidence": confidence.score,
                "is_reliable": confidence.is_reliable,
            },
        )
        return NonLLMResult(text=text, confidence=confidence, steps_executed=steps_executed, metadata=metadata)


def run_pipeline(text: str, threshold: float = 0.85) -> NonLLMResult:
    """Run the non-LLM extraction pipeline."""
    pipeline = NonLLMPipeline(threshold=threshold)
    return pipeline.run(text)
