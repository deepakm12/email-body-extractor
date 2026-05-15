# Architecture

## System Overview

The Email Body Extractor is structured as two independent services that communicate over HTTP:

```
+------------------+        HTTP        +-------------------+
|  Streamlit Web   |  --------------->  |   FastAPI Backend |
|  (port 8000)     |                    |   (port 3000)     |
+------------------+                    +-------------------+
                                               |
                              +----------------+----------------+
                              |                                 |
                    +---------v---------+           +-----------v--------+
                    |  Non-LLM Pipeline |           |   LLM Agent Flow   |
                    |  (deterministic)  |           |   (LangGraph)      |
                    +-------------------+           +--------------------+
                                                           |
                                             +-------------+------------+
                                             | OpenAI / Azure OpenAI    |
                                             | Anthropic / Gemini       |
                                             +--------------------------+
```

## Request Lifecycle

A POST to `/api/v1/extract` passes through these layers in order:

```
HTTP Request
    |
    v
FastAPI Route (src/app/api/v1/routes.py)
    |  - Validates ExtractionRequest via Pydantic
    |  - Delegates to ExtractionService
    v
ExtractionService (src/app/services/extraction_service.py)
    |  - Runs EmailPreprocessor (HTML parsing, .eml decoding, text normalization)
    |  - Delegates to ExtractionRouter
    v
ExtractionRouter (src/app/router/extraction_router.py)
    |  - Routes by mode: non_llm | llm | auto
    |
    +--[non_llm]--> NonLLMPipeline --> result
    |
    +--[llm]------> LLMWorkflow (LangGraph) --> result
    |
    +--[auto]-----> NonLLMPipeline
                        |
                        +-- is_reliable? --> return non_llm result
                        |
                        +-- not reliable --> LLMWorkflow
                                                |
                                                +-- success? --> return llm result
                                                |
                                                +-- failed  --> return non_llm result (fallback)
    |
    v
ExtractionService
    |  - Builds ExtractionResponse
    |  - Saves to history (extraction_history.json)
    v
HTTP Response
```

## Non-LLM Pipeline

The deterministic pipeline in `src/app/non_llm/` runs three sequential cleaning stages followed by confidence scoring.

### Preprocessing (before the pipeline)

`EmailPreprocessor` runs first and is responsible for format normalization:

| Step | What it does |
|------|--------------|
| EML parsing | Decodes `.eml` bytes, extracts `text/plain` and `text/html` parts |
| HTML cleaning | Strips script/style/iframe tags, Gmail quote divs, Outlook forward blocks, HR separators |
| HTML-to-text | Converts remaining HTML to readable plain text via BeautifulSoup |
| Apple Mail quotes | Removes `On <date> ... wrote:` patterns |
| Text normalization | NFC unicode normalization, null-byte removal, CRLF normalization, collapse whitespace |
| Disclaimer removal | Strips common legal boilerplate matching CONFIDENTIALITY/DISCLAIMER patterns |

### Pipeline Stages

```
Raw text (normalized)
    |
    v
[1] ReplyRemover  (src/app/non_llm/reply_remover.py)
    |   - Pass 1: email-reply-parser library (fragment-based quote detection)
    |   - Pass 2: Remove blockquoted lines (lines starting with >)
    |   - Pass 3: Regex patterns for Gmail, Outlook, Apple Mail, Yahoo, Thunderbird headers
    v
[2] SignatureRemover  (src/app/non_llm/signature_remover.py)
    |   - Primary: Talon library (ML-based signature boundary detection)
    |   - Fallback: regex for --, Best/Regards closings, "Sent from my ..." device lines
    v
[3] DisclaimerRemover  (src/app/non_llm/disclaimer_remover.py)
    |   - Regex patterns for legal disclaimers and confidentiality notices
    v
[4] ConfidenceScorer  (src/app/non_llm/confidence_scorer.py)
    |   - length_score  (weight 0.20): penalizes very short (<50 chars) or very long (>10k) text
    |   - noise_score   (weight 0.30): penalizes HTML tags, emails, URLs, phone numbers, etc.
    |   - quality_score (weight 0.20): rewards sentence endings, greeting phrases, common words
    |   - readability_score (weight 0.15): checks special-char ratio and word density
    |   - completeness_score (weight 0.15): checks final punctuation and sentence integrity
    |   - is_reliable = final_score >= CONFIDENCE_THRESHOLD (default 0.85)
    v
NonLLMResult { text, confidence, steps_executed, metadata }
```

Each stage returns `(cleaned_text, was_modified)`. A step is listed in `steps_executed` only if it changed the text.

## LangGraph Workflow (LLM Agent Flow)

The LLM flow uses a LangGraph `StateGraph` to orchestrate four agents. The state machine handles the cleanup-validation loop automatically.

```
Initial State: { content, current_text, cleanup_iterations=0, ... }

         +------------------+
         |  extraction_node |   ExtractionAgent
         |  Prompt: extract |   Returns: extracted_text (JSON)
         +--------+---------+
                  |
         +--------v---------+
         |   cleanup_node   |   CleanupAgent
         |  Prompt: clean   |   Returns: cleaned_text (JSON)
         +--------+---------+
                  |
         +--------v---------+
         | validation_node  |   ValidationAgent
         |  Prompt: validate|   Returns: is_valid, issues, suggested_fix
         +--------+---------+
                  |
         +--------v---------+
         |  should_continue |   Conditional edge
         | is_valid? or     |
         | iterations >= 2? |
         +--+----------+----+
            |          |
       [valid /    [invalid &
      max iter]    iter < 2]
            |          |
            |          +-----> cleanup_node (retry, max 2 times)
            |
   +--------v---------+
   |  confidence_node |   ConfidenceAgent
   |  Prompt: score   |   Returns: confidence_score (0.0-1.0)
   +--------+---------+
            |
           END

Final State: { final_text, confidence_score, trace, cleanup_iterations }
```

**Sequential fallback**: If LangGraph itself raises an exception (e.g., dependency conflict), the workflow automatically retries by running agents sequentially in a `for` loop with the same logic.

### Agents

Each agent follows the Template Method pattern (abstract base class `Agent`):

| Agent | Class | Input | Output JSON key | Role |
|-------|-------|-------|-----------------|------|
| extraction | `ExtractionAgent` | raw email text | `extracted_message` | Strips all reply chains, signatures, and context |
| cleanup | `CleanupAgent` | extracted text | `cleaned_message` | Removes formatting artifacts, normalizes whitespace |
| validation | `ValidationAgent` | cleaned text | `is_valid`, `issues`, `suggested_fix` | Quality gate; triggers re-cleanup if issues found |
| confidence | `ConfidenceAgent` | final text | `confidence_score` | Assigns a 0.0-1.0 reliability score |

All agents use exponential-backoff retries (up to 3 attempts, 1-10 second wait) via `tenacity`.

## Auto Mode Decision Logic

```
POST /extract  { mode: "auto" }
    |
    v
Run NonLLMPipeline
    |
    +-- confidence.is_reliable == True  (score >= threshold)
    |       |
    |       v
    |   Return non-LLM result
    |   flow_used = "non_llm (auto mode)"
    |
    +-- confidence.is_reliable == False (score < threshold)
            |
            v
        Run LLMWorkflow
            |
            +-- success == True
            |       |
            |       v
            |   Return LLM result
            |   flow_used = "llm (auto mode)"
            |
            +-- success == False  (LLM error / no API key)
                    |
                    v
                Return non-LLM result as fallback
                flow_used = "non_llm (auto mode, llm failed)"
```

The `flow_used` field in the response tells you exactly which path was taken.

## Module Structure

```
src/
└── app/
    ├── main.py                    # FastAPI app factory, CORS, lifespan
    ├── api/
    │   ├── router.py              # Mounts /api/v1 prefix
    │   └── v1/
    │       └── routes.py          # All six endpoint handlers
    ├── config/
    │   └── settings.py            # Pydantic-settings AppSettings, get_settings()
    ├── models/
    │   └── schemas.py             # ExtractionRequest, ExtractionResponse, etc.
    ├── common/
    │   ├── preprocessing.py       # EmailPreprocessor (HTML, EML, normalization)
    │   ├── exceptions.py          # EmailExtractionError, ProviderNotConfiguredError, etc.
    │   └── logging_config.py      # Structured JSON logger setup
    ├── non_llm/
    │   ├── base.py                # TextCleaner ABC
    │   ├── pipeline.py            # NonLLMPipeline orchestrator
    │   ├── reply_remover.py       # ReplyRemover (email-reply-parser + regex)
    │   ├── signature_remover.py   # SignatureRemover (Talon + regex fallback)
    │   ├── disclaimer_remover.py  # DisclaimerRemover
    │   └── confidence_scorer.py   # ConfidenceScorer (5-factor weighted scoring)
    ├── llm_flow/
    │   ├── agents.py              # ExtractionAgent, CleanupAgent, ValidationAgent, ConfidenceAgent
    │   └── workflow.py            # LLMWorkflow (LangGraph state machine + sequential fallback)
    ├── providers/
    │   ├── base.py                # BaseLLMProvider ABC (invoke, stream, is_configured)
    │   ├── factory.py             # get_provider(), list_available_providers()
    │   ├── openai_provider.py     # OpenAIProvider
    │   ├── azure_openai_provider.py  # AzureOpenAIProvider
    │   ├── anthropic_provider.py  # AnthropicProvider (via OpenAI-compatible endpoint)
    │   └── gemini_provider.py     # GeminiProvider (via OpenAI-compatible endpoint)
    ├── router/
    │   └── extraction_router.py   # ExtractionRouter (routes by mode)
    └── services/
        ├── extraction_service.py  # ExtractionService (orchestrates preprocessing + routing)
        └── history_service.py     # HistoryRepository (extraction_history.json, max 1000)

src/web/
└── main.py                        # Streamlit UI

tests/
├── conftest.py
├── test_api.py
├── test_non_llm_pipeline.py
├── test_preprocessing.py
├── test_providers.py
├── test_samples.py
├── test_schemas.py
├── test_settings.py
└── test_history_service.py
```
