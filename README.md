# Email Body Extraction Platform

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)

A hybrid platform that extracts the **latest meaningful message** from email threads in plain text, HTML, and `.eml` format. It combines a fast deterministic NLP pipeline with an agentic AI workflow (LangGraph) for cases where simple heuristics fall short.

In **auto mode** (the default), the deterministic pipeline runs first. If its confidence score meets the threshold, you get an instant result with no LLM call. Only when the result is uncertain does it escalate to the LLM agent flow — giving you accuracy when you need it and speed when you don't.

## Features

- **Multi-format support** — plain text, HTML (Gmail, Outlook, Apple Mail threading), and `.eml` files
- **Hybrid extraction pipeline** — deterministic NLP first, LLM agents as fallback in auto mode
- **Streaming API** — Server-Sent Events for real-time LLM token output
- **4 LLM providers** — OpenAI, Azure OpenAI, Anthropic, and Google Gemini (all via OpenAI-compatible SDK)
- **Confidence scoring** — 5-factor weighted score; `is_reliable` flag drives auto-mode routing
- **Extraction history** — persisted to `extraction_history.json`, queryable via API
- **Streamlit UI** — browser-based interface with provider selector, streaming toggle, and history panel
- **Docker ready** — single `docker-compose up` starts both API and UI

## Quick Start (Docker)

```bash
# 1. Clone
git clone <repo-url>
cd email-body-extractor

# 2. Configure (set at least one LLM provider API key)
cp .env.example .env

# 3. Start
docker-compose up
```

- API: `http://localhost:3000`
- Web UI: `http://localhost:8000`
- Interactive API docs: `http://localhost:3000/docs`

## Architecture

```
                          POST /api/v1/extract
                                  |
                     +------------v-----------+
                     |    EmailPreprocessor   |
                     | HTML parse / EML decode|
                     | normalize / disclaimers|
                     +------------+-----------+
                                  |
                     +------------v-----------+
                     |    ExtractionRouter    |
                     +---+--------+-------+---+
                         |        |       |
                    non_llm      llm    auto
                         |        |       |
          +--------------+        |   +---v--------------------------+
          |                       |   | Run NonLLM pipeline          |
          v                       |   | confidence >= 0.85?          |
  +--------------+                |   +---+------------------+-------+
  | NonLLM       |                |       |                  |
  | Pipeline     |                |      yes                 no
  |              |                |       |                  |
  | ReplyRemover |                |   return result      Run LLM flow
  | SigRemover   |                |                          |
  | DisclRemover |                |                   success? -> return LLM result
  | Confidence   |                |                   failed?  -> return nonLLM fallback
  +--------------+    +-----------v-----------+
                      |   LangGraph Workflow  |
                      | extraction -> cleanup |
                      | -> validation -> conf |
                      | (re-cleanup loop x2)  |
                      +-----------------------+
```

## API Usage

### Health check

```bash
curl http://localhost:3000/api/v1/health
```

```json
{"status": "healthy", "version": "1.0.0"}
```

### Extract email body

```bash
curl -X POST http://localhost:3000/api/v1/extract \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Let us sync tomorrow.\n\nOn Mon wrote:\n> Sounds good.",
    "mode": "auto"
  }'
```

```json
{
  "success": true,
  "data": {
    "latest_message": "Let us sync tomorrow.",
    "confidence": 0.912,
    "flow_used": "non_llm (auto mode)",
    "metadata": {...}
  },
  "error": null,
  "agent_trace": null
}
```

### Extract with LLM and specific provider

```bash
curl -X POST http://localhost:3000/api/v1/extract \
  -H "Content-Type: application/json" \
  -d '{
    "content": "<html>..complex thread..</html>",
    "mode": "llm",
    "provider": "anthropic"
  }'
```

### Stream LLM tokens

```bash
curl -N -X POST http://localhost:3000/api/v1/extract/stream \
  -H "Content-Type: application/json" \
  -d '{"content": "Can you call me?", "mode": "llm"}'
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | Default provider: `openai`, `azure_openai`, `anthropic`, `gemini` |
| `MODEL_NAME` | `gpt-4.1-mini` | Model name for the selected provider |
| `TEMPERATURE` | `0` | LLM sampling temperature |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `AZURE_OPENAI_API_KEY` | — | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | — | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT` | — | Azure OpenAI deployment name |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `CONFIDENCE_THRESHOLD` | `0.85` | Non-LLM score threshold for auto-mode LLM fallback |
| `API_BASE_URL` | `http://localhost:3000` | URL the Streamlit UI uses to reach the API |
| `DEBUG` | `false` | FastAPI debug mode + LangGraph verbose logging |
| `LOG_LEVEL` | `INFO` | `INFO` or `DEBUG` |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |

See [docs/configuration.md](docs/configuration.md) for provider-specific setup and confidence threshold tuning guidance.

## Development Setup

**Requirements**: Python 3.10.x, Poetry >= 2.0.0

```bash
# Install dependencies
poetry install

# Copy environment file
cp .env.example .env

# Run the API (from project root)
cd src && poetry run uvicorn app.main:app --host 0.0.0.0 --port 3000 --reload

# Run the web UI (separate terminal, from project root)
poetry run streamlit run src/web/main.py --server.port 8000
```

## Testing

```bash
# Run all tests
PYTHONPATH=src poetry run pytest

# With coverage
PYTHONPATH=src poetry run pytest --cov=app --cov-report=term-missing

# Run live tests (calls real LLM APIs, costs tokens)
PYTHONPATH=src poetry run pytest -m live
```

## Project Structure

```
email-body-extractor/
├── .env.example
├── docker-compose.yml
├── Dockerfile                  # API service
├── Dockerfile.web              # Streamlit UI
├── pyproject.toml
├── docs/
│   ├── architecture.md         # System design, pipeline stages, LangGraph state machine
│   ├── api.md                  # All endpoints, request/response schemas, curl examples
│   ├── configuration.md        # All env vars, provider setup, threshold tuning
│   └── development.md          # Local setup, testing, code quality, extending the system
├── src/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/v1/routes.py    # 6 endpoint handlers
│   │   ├── config/settings.py  # Pydantic-settings
│   │   ├── models/schemas.py   # Request/response models
│   │   ├── common/             # Preprocessing, logging, exceptions
│   │   ├── non_llm/            # Deterministic pipeline
│   │   ├── llm_flow/           # LangGraph agents + workflow
│   │   ├── providers/          # OpenAI, Azure, Anthropic, Gemini
│   │   ├── router/             # Auto/non-LLM/LLM routing
│   │   └── services/           # ExtractionService, HistoryService
│   └── web/
│       └── main.py             # Streamlit UI
└── tests/
```

## License

MIT License — see [LICENSE](LICENSE) for details.
