# Development Guide

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.10.x (exactly) | `pyproject.toml` pins `>=3.10,<3.11`. Use pyenv or `.tool-versions` |
| Poetry | >=2.0.0, <3.0.0 | Package and dependency manager |
| Docker + Compose | any recent | Only needed for the containerized workflow |

The project ships a `.tool-versions` file for [asdf](https://asdf-vm.com/) / [mise](https://mise.jdx.dev/). If you use either tool, run `asdf install` or `mise install` to get the pinned Python version automatically.

## Local Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd email-body-extractor

# 2. Install dependencies (creates a .venv in the project root via poetry.toml)
poetry install

# 3. Copy and edit the environment file
cp .env.example .env
# Open .env and set at least one LLM provider API key
```

`poetry.toml` sets `virtualenvs.in-project = true`, so the virtualenv lands at `.venv/` inside the project directory.

## Running the API Server

```bash
# Run from the project root — PYTHONPATH is handled by the Dockerfile / poetry config
cd src
poetry run uvicorn app.main:app --host 0.0.0.0 --port 3000 --reload
```

The `--reload` flag enables hot-reload on file changes. The API will be available at:

- API: `http://localhost:3000`
- Interactive docs: `http://localhost:3000/docs`

## Running the Streamlit UI

Open a second terminal:

```bash
poetry run streamlit run src/web/main.py --server.port 8000
```

The UI connects to the API at the URL specified by `API_BASE_URL` in `.env` (default: `http://localhost:3000`).

## Running Tests

```bash
# Run all tests (excludes live API tests by default)
PYTHONPATH=src poetry run pytest

# Run with verbose output
PYTHONPATH=src poetry run pytest -v

# Run with coverage report
PYTHONPATH=src poetry run pytest --cov=app --cov-report=term-missing

# Run a specific test file
PYTHONPATH=src poetry run pytest tests/test_non_llm_pipeline.py

# Run live tests (hits real OpenAI API — costs tokens)
PYTHONPATH=src poetry run pytest -m live
```

`PYTHONPATH=src` is required because the package is in `src/app/` and the test files import from `app.*` directly. The `pyproject.toml` also sets `pythonpath = ["src"]` for pytest, so `poetry run pytest` handles it automatically without needing the prefix.

### Test Markers

| Marker | Description |
|--------|-------------|
| (none) | Default: all unit and integration tests without external calls |
| `live` | Tests that call real LLM APIs. Excluded by default. Run with `-m live`. |

## Code Quality Tools

All tools are configured in `pyproject.toml` with a 120-character line length.

```bash
# Format code with black
poetry run black src/ tests/

# Sort imports
poetry run isort src/ tests/

# Lint with ruff (fast, covers most flake8 rules)
poetry run ruff check src/ tests/

# Lint with flake8 (see .flake8 for config)
poetry run flake8 src/ tests/

# Type-check with mypy (strict mode)
poetry run mypy src/
```

Run all checks in sequence:

```bash
poetry run black src/ tests/ && \
poetry run isort src/ tests/ && \
poetry run ruff check src/ tests/ && \
poetry run mypy src/
```

### mypy notes

`mypy` is configured with `strict = true`. Third-party stubs for `talon`, `email_reply_parser`, and `google.*` are absent; these are silenced via `[[tool.mypy.overrides]]` in `pyproject.toml`.

## Docker Development

### Start both services

```bash
# Build and start API + web UI
docker-compose up --build

# Start in detached mode
docker-compose up -d --build
```

Services:
- API: `http://localhost:3000`
- Web UI: `http://localhost:8000`

Both services load environment variables from `.env`. The web container sets `API_BASE_URL=http://api:3000` internally so that it reaches the API over the Docker network by service name.

### Start only the API

```bash
docker-compose up api
```

### Rebuild after dependency changes

```bash
docker-compose build --no-cache
docker-compose up
```

### View logs

```bash
docker-compose logs -f api
docker-compose logs -f web
```

### Stop and remove containers

```bash
docker-compose down
```

## Project Structure

```
email-body-extractor/
├── .env.example              # Template for environment variables
├── .flake8                   # flake8 configuration
├── .tool-versions            # asdf/mise Python version pin
├── docker-compose.yml        # Multi-service Docker Compose definition
├── Dockerfile                # API service container (Python 3.10, uvicorn)
├── Dockerfile.web            # Web UI container (Streamlit)
├── extraction_history.json   # Runtime history file (auto-created, gitignored)
├── poetry.lock               # Locked dependency tree
├── poetry.toml               # Poetry local settings (in-project venv)
├── pyproject.toml            # Project metadata, dependencies, tool config
├── docs/                     # This documentation
│   ├── architecture.md
│   ├── api.md
│   ├── configuration.md
│   └── development.md
├── src/
│   ├── app/                  # FastAPI application package
│   │   ├── main.py           # App factory
│   │   ├── api/v1/routes.py  # Endpoint handlers
│   │   ├── config/settings.py
│   │   ├── models/schemas.py
│   │   ├── common/           # Shared utilities (preprocessing, logging, exceptions)
│   │   ├── non_llm/          # Deterministic pipeline (reply, signature, disclaimer, confidence)
│   │   ├── llm_flow/         # LangGraph agents and workflow
│   │   ├── providers/        # LLM provider abstractions (OpenAI, Azure, Anthropic, Gemini)
│   │   ├── router/           # Extraction routing logic (auto/non_llm/llm)
│   │   └── services/         # ExtractionService, HistoryService
│   └── web/
│       └── main.py           # Streamlit UI
└── tests/                    # pytest test suite
```

## Adding a New LLM Provider

1. Create `src/app/providers/<name>_provider.py` implementing `BaseLLMProvider` from `src/app/providers/base.py`. Implement `invoke()`, `stream()`, and `is_configured()`.
2. Add the new provider name to `LlmProviderType` in `src/app/config/settings.py`.
3. Add a default model entry to `_DEFAULT_MODELS` in `src/app/providers/factory.py`.
4. Add a `case` for your new type in both `_resolve_provider_name()` and `_create_provider_instance()` in `factory.py`.
5. Add the corresponding API key variable to `AppSettings` in `settings.py` and to `.env.example`.
6. Add a `case` in `AppSettings.provider_api_key` to return the new key.
7. Write a test in `tests/test_providers.py`.

## Extending the Non-LLM Pipeline

Each pipeline stage implements `TextCleaner` from `src/app/non_llm/base.py`:

```python
class TextCleaner(ABC):
    step_name: str

    @abstractmethod
    def clean(self, text: str) -> tuple[str, bool]:
        """Return (cleaned_text, was_modified)."""
```

To add a new stage:

1. Create a new file in `src/app/non_llm/` and implement `TextCleaner`.
2. Add an instance of it to the `steps` list in `NonLLMPipeline.__init__()` in `pipeline.py`, or inject it via the `steps` constructor argument.
3. Write tests for the new stage.
