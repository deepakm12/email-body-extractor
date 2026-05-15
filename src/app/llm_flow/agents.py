"""LLM Agent definitions for the extraction workflow."""

import re
from abc import ABC, abstractmethod
from textwrap import dedent
from typing import ClassVar

from pydantic import BaseModel, Field
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.common.exceptions import RetriableProviderError
from app.common.logging_config import logger
from app.providers.base import BaseLLMProvider

_MAX_RETRIES = 3
_RETRY_WAIT = wait_exponential(multiplier=1, min=1, max=10)
_RETRY_POLICY = retry_if_exception_type(RetriableProviderError)

# System messages for each agent
EXTRACTION_SYSTEM = dedent("""\
    You are an expert email parser. Your task is to extract ONLY the latest/real
    email message from the provided content.

    Rules:
    - Remove all quoted replies, forwarded messages, and previous email chains
    - Remove all signatures and legal disclaimers
    - Return ONLY the actual latest message content
    - Preserve the original language and meaning
    - If the email contains only a reply chain with no new content, return an empty string
    - Do not add any commentary, headers, or explanations
    - Do not summarize or paraphrase - extract verbatim

    Respond in JSON format:
    {
        "extracted_message": "the extracted message content, or empty string if no new content"
    }""")

CLEANUP_SYSTEM = dedent("""\
    You are an expert text cleaner. Your task is to clean up the provided email message.

    Rules:
    - Remove any remaining formatting artifacts (HTML tags, special characters sequences)
    - Fix any broken sentences or words
    - Normalize whitespace (no excessive blank lines, no leading/trailing whitespace)
    - Remove any remaining "Sent from my..." device signatures
    - Remove any URLs that appear to be tracking pixels or unsubscribe links
    - Keep the message natural and readable
    - Preserve all meaningful content including necessary URLs
    - Do not add any commentary or explanations

    Respond in JSON format:
    {
        "cleaned_message": "the cleaned message content"
    }""")

VALIDATION_SYSTEM = dedent("""\
    You are an expert quality validator. Your task is to validate an extracted email message.

    Check for:
    1. Completeness - is the message complete (not cut off mid-sentence)?
    2. Coherence - does it make sense as a standalone message?
    3. Noise - is there any remaining quoted content, signatures, or disclaimers?
    4. Quality - is it a real, meaningful message (not just "thanks" or "ok")?

    Respond in JSON format:
    {
        "is_valid": true/false,
        "issues": ["list any problems found"],
        "suggested_fix": "description of what needs to be fixed, or empty if valid"
    }

    Be strict - if there are issues, flag them.""")

CONFIDENCE_SYSTEM = dedent("""\
    You are an expert at assessing text quality. Rate the reliability of this extracted email message.

    Consider:
    - How complete and coherent is the message?
    - Is there any remaining noise or artifacts?
    - Is the message meaningful and substantive?
    - How confident are you that this represents the actual latest email?

    Score ranges:
    - 0.9-1.0 = Excellent, clean extraction
    - 0.7-0.89 = Good, minor issues
    - 0.5-0.69 = Acceptable, some noise may remain
    - Below 0.5 = Poor, significant problems

    Respond in JSON format:
    {
        "confidence_score": 0.95
    }""")


class ExtractionOutput(BaseModel):
    """Pydantic DTO for extraction agent response."""

    extracted_message: str


class CleanupOutput(BaseModel):
    """Pydantic DTO for cleanup agent response."""

    cleaned_message: str


class ValidationOutput(BaseModel):
    """Pydantic DTO for validation agent response."""

    is_valid: bool
    issues: list[str] = Field(default_factory=list)
    suggested_fix: str = ""


class ConfidenceOutput(BaseModel):
    """Pydantic DTO for confidence agent response."""

    confidence_score: float = Field(ge=0.0, le=1.0)


class AgentResult(BaseModel):
    """Result from an agent execution."""

    output: str
    agent_name: str
    success: bool
    metadata: dict[str, object] = Field(default_factory=dict)
    execution_time_ms: float | None = None


def _clean_json_response(text: str) -> str:
    """Clean LLM response to extract valid JSON."""
    # Remove markdown code blocks
    text = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL)
    # Extract JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group()
    return text


class Agent(ABC):
    """Template Method base class for LLM extraction agents."""

    agent_name: ClassVar[str] = ""

    @abstractmethod
    def _build_prompt(self, content: str) -> str:
        """Build the prompt for this agent."""

    @abstractmethod
    def _system_message(self) -> str:
        """Return the system message for this agent."""

    @abstractmethod
    def _parse_response(self, result: str, original_content: str) -> AgentResult:
        """Parse the LLM response into an AgentResult."""

    def _on_exception(self, content: str, error: Exception) -> AgentResult:
        logger.error(f"{self.agent_name.title()} agent failed: {error}")
        return AgentResult(output=content, agent_name=self.agent_name, success=False, metadata={"error": str(error)})

    def _invoke_with_retry(self, provider: BaseLLMProvider, prompt: str) -> str:
        """Invoke the provider with exponential-backoff retries."""
        for attempt in Retrying(stop=stop_after_attempt(_MAX_RETRIES), wait=_RETRY_WAIT, retry=_RETRY_POLICY, reraise=True):
            with attempt:
                attempt_number = attempt.retry_state.attempt_number
                if attempt_number > 1:
                    logger.warning("%s agent retry %d/%d", self.agent_name, attempt_number, _MAX_RETRIES)
                return provider.invoke(prompt, system_message=self._system_message())
        raise RuntimeError("unreachable")  # keeps type-checkers happy

    def run(self, provider: BaseLLMProvider, content: str) -> AgentResult:
        """Run the agent using the template method pattern."""
        logger.debug(f"Running {self.agent_name} agent")
        prompt = self._build_prompt(content)
        try:
            result = self._invoke_with_retry(provider, prompt)
            return self._parse_response(result=result, original_content=content)
        except Exception as e:
            return self._on_exception(content, e)


class ExtractionAgent(Agent):
    """Extracts the latest message from raw email content."""

    agent_name = "extraction"

    def _build_prompt(self, content: str) -> str:
        return f"Extract the latest email message from this content:\n\n{content}"

    def _system_message(self) -> str:
        return EXTRACTION_SYSTEM

    def _parse_response(self, result: str, original_content: str) -> AgentResult:
        cleaned = _clean_json_response(result)
        try:
            output = ExtractionOutput.model_validate_json(cleaned)
        except Exception:
            output = ExtractionOutput(extracted_message=result.strip())
        return AgentResult(output=output.extracted_message, agent_name=self.agent_name, success=True, metadata={})


class CleanupAgent(Agent):
    """Removes noise and formatting artifacts from extracted text."""

    agent_name = "cleanup"

    def _build_prompt(self, content: str) -> str:
        return f"Clean up this email message:\n\n{content}"

    def _system_message(self) -> str:
        return CLEANUP_SYSTEM

    def _parse_response(self, result: str, original_content: str) -> AgentResult:
        cleaned = _clean_json_response(result)
        try:
            output = CleanupOutput.model_validate_json(cleaned)
        except Exception:
            output = CleanupOutput(cleaned_message=result.strip())
        return AgentResult(output=output.cleaned_message, agent_name=self.agent_name, success=True, metadata={})


class ValidationAgent(Agent):
    """Validates extraction quality and flags issues."""

    agent_name = "validation"

    def _build_prompt(self, content: str) -> str:
        return f"Validate this extracted email message:\n\n{content}"

    def _system_message(self) -> str:
        return VALIDATION_SYSTEM

    def _parse_response(self, result: str, original_content: str) -> AgentResult:
        cleaned = _clean_json_response(result)
        try:
            validation = ValidationOutput.model_validate_json(cleaned)
        except Exception:
            validation = ValidationOutput(
                is_valid="invalid" not in result.lower() and "false" not in result.lower(),
                issues=["Could not parse structured validation"],
            )
        return AgentResult(
            output=original_content,
            agent_name=self.agent_name,
            success=validation.is_valid,
            metadata={"validation": validation.model_dump(), "raw_response": result},
        )

    def _on_exception(self, content: str, error: Exception) -> AgentResult:
        logger.error(f"Validation agent failed: {error}")
        return AgentResult(
            output=content,
            agent_name=self.agent_name,
            success=True,  # Allow pipeline to continue despite validation failure
            metadata={"error": str(error)},
        )


class ConfidenceAgent(Agent):
    """Assigns a reliability score to the extracted message."""

    agent_name = "confidence"

    def _build_prompt(self, content: str) -> str:
        return f"Rate the quality of this extracted email message (0.0-1.0):\n\n{content}"

    def _system_message(self) -> str:
        return CONFIDENCE_SYSTEM

    def _parse_response(self, result: str, original_content: str) -> AgentResult:
        cleaned = _clean_json_response(result)
        try:
            output = ConfidenceOutput.model_validate_json(cleaned)
        except Exception:
            score_match = re.search(r"(\d+\.?\d*)", result.replace(",", "."))
            if score_match:
                score = float(score_match.group())
                if score > 1.0:
                    score = score / 100.0
                score = max(0.0, min(1.0, score))
            else:
                score = 0.5
            output = ConfidenceOutput(confidence_score=round(score, 3))
        return AgentResult(
            output=original_content,
            agent_name=self.agent_name,
            success=True,
            metadata={"confidence_score": output.confidence_score},
        )

    def _on_exception(self, content: str, error: Exception) -> AgentResult:
        logger.error(f"Confidence agent failed: {error}")
        return AgentResult(
            output=content,
            agent_name=self.agent_name,
            success=True,
            metadata={"confidence_score": 0.5, "error": str(error)},
        )


def cleanup_agent(provider: BaseLLMProvider, content: str) -> AgentResult:
    """Run the cleanup agent and return its result."""
    return CleanupAgent().run(provider, content)


def confidence_agent(provider: BaseLLMProvider, content: str) -> AgentResult:
    """Run the confidence agent and return its result."""
    return ConfidenceAgent().run(provider, content)
