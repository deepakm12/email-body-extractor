"""LangGraph-based LLM extraction workflow."""

import time
from dataclasses import dataclass
from typing import TypedDict, cast

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.config.settings import AppSettings, get_settings
from app.llm_flow.agents import CleanupAgent, ConfidenceAgent, ExtractionAgent, ValidationAgent
from app.providers.base import BaseLLMProvider
from app.common.logging_config import logger


class _WorkflowState(TypedDict):
    """Workflow state for LangGraph execution."""

    content: str
    current_text: str
    extracted_text: str
    cleaned_text: str
    final_text: str
    confidence_score: float
    is_valid: bool
    validation_result: dict[str, object]
    trace: list[dict[str, object]]
    cleanup_iterations: int


@dataclass
class LLMFlowResult:
    """Result of the LLM agent flow."""

    text: str
    confidence: float
    agent_trace: list[dict[str, object]]
    iterations: int = 0
    success: bool = True


class LLMWorkflow:
    """Orchestrates LLM agents for email body extraction using LangGraph or sequential fallback."""

    _provider: BaseLLMProvider
    _max_iterations: int
    _extraction: ExtractionAgent
    _cleanup: CleanupAgent
    _settings: AppSettings
    _validation: ValidationAgent
    _confidence: ConfidenceAgent

    def __init__(self, provider: BaseLLMProvider, max_iterations: int = 3) -> None:
        self._provider = provider
        self._max_iterations = max_iterations
        self._settings = get_settings()
        self._extraction = ExtractionAgent()
        self._cleanup = CleanupAgent()
        self._validation = ValidationAgent()
        self._confidence = ConfidenceAgent()

    def run(self, content: str) -> LLMFlowResult:
        """Run the extraction workflow, using LangGraph if available."""
        logger.info("Starting LLM extraction flow")
        return self._run_with_langgraph(content)

    def _create_langgraph_workflow(self) -> CompiledStateGraph[_WorkflowState]:
        """Build and compile the LangGraph state machine."""

        def extraction_node(state: _WorkflowState) -> _WorkflowState:
            """Extraction node: runs the extraction agent."""
            result = self._extraction.run(self._provider, state["content"])
            return {
                **state,
                "extracted_text": result.output,
                "current_text": result.output,
                "trace": state["trace"]
                + [
                    {
                        "agent": result.agent_name,
                        "success": result.success,
                        "output_preview": result.output[:200] if result.output else "",
                    }
                ],
            }

        def cleanup_node(state: _WorkflowState) -> _WorkflowState:
            """Cleanup node: runs the cleanup agent."""
            result = self._cleanup.run(self._provider, state["current_text"])
            return {
                **state,
                "cleaned_text": result.output,
                "current_text": result.output,
                "cleanup_iterations": state["cleanup_iterations"] + 1,
                "trace": state["trace"]
                + [
                    {
                        "agent": result.agent_name,
                        "success": result.success,
                        "output_preview": result.output[:200] if result.output else "",
                    }
                ],
            }

        def validation_node(state: _WorkflowState) -> _WorkflowState:
            """Validation node: runs the validation agent."""
            result = self._validation.run(self._provider, state["current_text"])
            return {
                **state,
                "validation_result": cast(dict[str, object], result.metadata.get("validation", {})),
                "is_valid": result.success,
                "trace": state["trace"]
                + [
                    {
                        "agent": result.agent_name,
                        "success": result.success,
                        "details": result.metadata.get("validation", {}),
                    }
                ],
            }

        def confidence_node(state: _WorkflowState) -> _WorkflowState:
            """Confidence node: runs the confidence agent."""
            result = self._confidence.run(self._provider, state["current_text"])
            return {
                **state,
                "confidence_score": cast(float, result.metadata.get("confidence_score", 0.5)),
                "final_text": state["current_text"],
                "trace": state["trace"]
                + [
                    {
                        "agent": result.agent_name,
                        "success": result.success,
                        "confidence": result.metadata.get("confidence_score"),
                    }
                ],
            }

        def should_continue(state: _WorkflowState) -> str:
            """Should we continue to cleanup or proceed to confidence evaluation?"""
            if state.get("is_valid", True):
                return "confidence"
            if state.get("cleanup_iterations", 0) >= 2:
                logger.warning("Max cleanup iterations reached, proceeding with confidence")
                return "confidence"
            return "cleanup"

        workflow = StateGraph(_WorkflowState)
        workflow.add_node("extraction", extraction_node)
        workflow.add_node("cleanup", cleanup_node)
        workflow.add_node("validation", validation_node)
        workflow.add_node("confidence", confidence_node)
        workflow.add_edge("extraction", "cleanup")
        workflow.add_edge("cleanup", "validation")
        workflow.add_conditional_edges(
            "validation",
            should_continue,
            {"cleanup": "cleanup", "confidence": "confidence"},
        )
        workflow.add_edge("confidence", END)
        workflow.set_entry_point("extraction")
        return workflow.compile(debug=self._settings.debug)

    def _run_with_langgraph(self, content: str) -> LLMFlowResult:
        """Execute flow using LangGraph."""
        workflow = self._create_langgraph_workflow()
        initial_state: _WorkflowState = {
            "content": content,
            "current_text": content,
            "extracted_text": "",
            "cleaned_text": "",
            "final_text": "",
            "confidence_score": 0.0,
            "is_valid": False,
            "validation_result": {},
            "trace": [],
            "cleanup_iterations": 0,
        }
        try:
            result_state = workflow.invoke(initial_state)
            return LLMFlowResult(
                text=result_state.get("final_text", ""),
                confidence=result_state.get("confidence_score", 0.0),
                agent_trace=result_state.get("trace", []),
                iterations=result_state.get("cleanup_iterations", 0),
                success=True,
            )
        except Exception as e:
            logger.error(f"LangGraph workflow failed: {e}")
            return self._run_sequential(content)

    def _run_sequential(self, content: str) -> LLMFlowResult:
        """Execute agents sequentially as fallback."""
        start_time = time.time()
        trace: list[dict[str, object]] = []
        current_text = content

        # Step 1: Extraction
        result = self._extraction.run(self._provider, current_text)
        current_text = result.output
        trace.append(
            {
                "agent": result.agent_name,
                "success": result.success,
                "output_preview": result.output[:200] if result.output else "",
            }
        )

        # Steps 2-3: Cleanup and Validation (with loop)
        iterations = 0
        for _ in range(self._max_iterations):
            result = self._cleanup.run(self._provider, current_text)
            current_text = result.output
            trace.append(
                {
                    "agent": result.agent_name,
                    "success": result.success,
                    "output_preview": result.output[:200] if result.output else "",
                }
            )
            iterations += 1

            result = self._validation.run(self._provider, current_text)
            trace.append(
                {
                    "agent": result.agent_name,
                    "success": result.success,
                    "details": result.metadata.get("validation", {}),
                }
            )

            if result.success:
                break

            validation_data = cast(dict[str, object], result.metadata.get("validation", {}))
            suggested_fix = cast(str, validation_data.get("suggested_fix", ""))
            if suggested_fix:
                current_text = f"{current_text}\n\n[Fix needed: {suggested_fix}]"

        # Step 4: Confidence
        result = self._confidence.run(self._provider, current_text)
        confidence_score = cast(float, result.metadata.get("confidence_score", 0.5))
        trace.append(
            {
                "agent": result.agent_name,
                "success": result.success,
                "confidence": confidence_score,
            }
        )

        elapsed = time.time() - start_time
        logger.info(
            "LLM sequential flow complete",
            extra={"confidence": confidence_score, "iterations": iterations, "elapsed_ms": round(elapsed * 1000, 2)},
        )
        return LLMFlowResult(
            text=current_text,
            confidence=confidence_score,
            agent_trace=trace,
            iterations=iterations,
            success=True,
        )


def run_llm_flow(content: str, provider: BaseLLMProvider, max_iterations: int = 3) -> LLMFlowResult:
    """Run the LLM agent extraction flow."""
    workflow = LLMWorkflow(provider, max_iterations)
    return workflow.run(content)
