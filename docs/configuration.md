# Configuration

All configuration is loaded from environment variables. Copy `.env.example` to `.env` and fill in the values you need.

```bash
cp .env.example .env
```

## All Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `APP_NAME` | string | `Email Body Extractor` | Application display name |
| `APP_VERSION` | string | auto-detected | Version string (auto-read from package metadata) |
| `DEBUG` | boolean | `false` | Enable FastAPI debug mode and LangGraph verbose output |
| `LOG_LEVEL` | string | `INFO` | Log verbosity: `INFO` or `DEBUG` |
| `CORS_ORIGINS` | string | `*` | Comma-separated list of allowed CORS origins |
| `LLM_PROVIDER` | string | `openai` | Default LLM provider: `openai`, `azure_openai`, `anthropic`, `gemini` |
| `MODEL_NAME` | string | `gpt-4.1-mini` | Model name to use with the selected provider |
| `TEMPERATURE` | float | `0` | LLM sampling temperature (0 = deterministic) |
| `OPENAI_API_KEY` | string | — | OpenAI API key |
| `AZURE_OPENAI_API_KEY` | string | — | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | string | — | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT` | string | — | Azure OpenAI deployment name |
| `GEMINI_API_KEY` | string | — | Google Gemini API key |
| `ANTHROPIC_API_KEY` | string | — | Anthropic API key |
| `CONFIDENCE_THRESHOLD` | float | `0.85` | Non-LLM confidence threshold for auto mode fallback |
| `API_BASE_URL` | string | `http://localhost:3000` | URL the Streamlit UI uses to reach the API |

## Provider-Specific Configuration

### OpenAI

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
MODEL_NAME=gpt-4.1-mini
```

Any model available through the OpenAI API can be used. Recommended models:

- `gpt-4.1-mini` — fast and cost-efficient (default)
- `gpt-4.1` — higher accuracy for complex threads
- `gpt-4o-mini` — alternative fast option

### Azure OpenAI

Azure OpenAI requires three additional variables beyond the API key:

```env
LLM_PROVIDER=azure_openai
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
MODEL_NAME=gpt-4.1-mini
```

`AZURE_OPENAI_DEPLOYMENT` is the name of the deployment you created in Azure AI Studio. The `MODEL_NAME` value is passed for reference but the actual model is determined by your deployment. If either `AZURE_OPENAI_ENDPOINT` or `AZURE_OPENAI_DEPLOYMENT` are missing, the provider will refuse to initialize.

### Anthropic

Anthropic's API is accessed via its OpenAI-compatible endpoint. The SDK internally routes the request to `https://api.anthropic.com`.

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
MODEL_NAME=claude-haiku-4-5
```

Default model when `MODEL_NAME` is not set: `claude-haiku-4-5`

Other supported models:
- `claude-sonnet-4-5` — higher accuracy, slower
- `claude-opus-4-5` — highest accuracy

### Google Gemini

Gemini is accessed via its OpenAI-compatible endpoint.

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIza...
MODEL_NAME=gemini-2.5-flash-lite
```

Default model when `MODEL_NAME` is not set: `gemini-2.5-flash-lite`

Other options:
- `gemini-2.5-flash` — slightly higher accuracy

## Per-Request Provider Override

You can override the server's default provider on a per-request basis by setting the `provider` field in the request body. This is independent of `LLM_PROVIDER`. The override must still be backed by a configured API key in the environment.

```json
{
  "content": "...",
  "mode": "llm",
  "provider": "anthropic"
}
```

## Confidence Threshold Tuning

`CONFIDENCE_THRESHOLD` controls when `auto` mode falls back from the non-LLM pipeline to the LLM agent flow.

The non-LLM confidence score is a weighted combination of five factors:

| Factor | Weight | What it measures |
|--------|--------|-----------------|
| `length_score` | 0.20 | Penalizes text shorter than 50 chars or longer than 10,000 chars |
| `noise_score` | 0.30 | Penalizes remaining HTML tags, email addresses, URLs, quoted-reply markers |
| `quality_score` | 0.20 | Rewards sentence-ending punctuation, greeting phrases, common English words |
| `readability_score` | 0.15 | Penalizes high special-character ratio; rewards natural word density |
| `completeness_score` | 0.15 | Checks whether the text ends on a complete sentence |

**Tuning guidance:**

- `0.85` (default) — conservative: the non-LLM pipeline must produce a very clean result before skipping the LLM. Good for production when LLM cost is acceptable.
- `0.70` — balanced: accepts moderate-quality non-LLM output. Reduces LLM calls by 20-40% on well-formatted emails.
- `0.60` — aggressive: falls back to LLM only for clearly noisy results. Minimizes LLM usage and cost.
- `1.0` — always use LLM (non-LLM result will never be considered reliable in auto mode).
- `0.0` — never use LLM (non-LLM result always accepted; equivalent to forcing `mode: "non_llm"`).

## CORS Configuration

`CORS_ORIGINS` accepts a comma-separated list of allowed origins. The API uses `allow_methods=["*"]` and `allow_headers=["*"]` regardless of this setting.

```env
# Allow all origins (development default)
CORS_ORIGINS=*

# Single origin
CORS_ORIGINS=https://myapp.example.com

# Multiple origins
CORS_ORIGINS=https://app.example.com,https://admin.example.com
```

## Log Level Options

| Value | Effect |
|-------|--------|
| `INFO` | Default. Logs request routing, pipeline stage completion, and confidence scores |
| `DEBUG` | Adds per-stage character counts, regex match details, Talon signature previews, and LLM prompt construction |

Set `DEBUG=true` together with `LOG_LEVEL=DEBUG` to also enable FastAPI's internal debug mode and LangGraph's step-by-step state logging.
