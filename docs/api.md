# API Reference

All endpoints are served under the `/api/v1` prefix. The API is a standard FastAPI application with automatic OpenAPI docs at `/docs`.

## Endpoints

### GET /api/v1/health

Returns service health and version information.

**Response** `200 OK`

```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

**Example**

```bash
curl http://localhost:3000/api/v1/health
```

---

### POST /api/v1/extract

Extract the latest message from an email thread.

**Request Body** (`application/json`)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `content` | string | yes | — | Email content: plain text, HTML, or `.eml` decoded as string. Must be non-empty. |
| `mode` | string | no | `"auto"` | Extraction mode: `non_llm`, `llm`, or `auto` |
| `provider` | string \| null | no | `null` | LLM provider override. One of: `openai`, `azure_openai`, `anthropic`, `gemini`. `null` uses the server default (`LLM_PROVIDER` env var). |
| `is_eml` | boolean | no | `false` | Set to `true` when `content` is an `.eml` file decoded to a string |

**Response** `200 OK`

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether extraction succeeded |
| `data` | object \| null | `ExtractionResult` on success, `null` on failure |
| `data.latest_message` | string | Extracted latest email message body |
| `data.confidence` | float | Confidence score, 0.0 to 1.0 |
| `data.flow_used` | string | Which pipeline ran: see [flow_used values](#flow_used-values) |
| `data.metadata` | object | Pipeline-specific metadata (stage lengths, scores, provider info, etc.) |
| `error` | string \| null | Error description when `success` is `false` |
| `agent_trace` | array \| null | Ordered list of agent execution steps (only present when LLM flow ran) |

#### `flow_used` values

| Value | Meaning |
|-------|---------|
| `non_llm` | Deterministic pipeline was used directly |
| `llm` | LLM agent flow was used directly |
| `non_llm (auto mode)` | Auto mode: non-LLM result was reliable enough |
| `llm (auto mode)` | Auto mode: non-LLM confidence was low, LLM was used |
| `non_llm (auto mode, llm failed)` | Auto mode: LLM failed, fell back to non-LLM result |

#### `agent_trace` entry fields

Each entry in `agent_trace` represents one agent execution step:

| Field | Type | Description |
|-------|------|-------------|
| `agent` | string | Agent name: `extraction`, `cleanup`, `validation`, `confidence` |
| `success` | boolean | Whether the agent completed without error |
| `output_preview` | string | First 200 chars of agent output (extraction and cleanup steps) |
| `details` | object | Validation details: `is_valid`, `issues`, `suggested_fix` (validation step only) |
| `confidence` | float | Confidence score (confidence step only) |

**Error Response** `422 Unprocessable Entity`

Returned when extraction fails (e.g., empty content after preprocessing, provider not configured):

```json
{
  "detail": "Extraction failed"
}
```

**Example: plain text, auto mode**

```bash
curl -X POST http://localhost:3000/api/v1/extract \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Hi, can we reschedule the meeting?\n\nOn Mon, Jan 1 2024 wrote:\n> Sure, let me know.",
    "mode": "auto"
  }'
```

**Example: HTML email, LLM mode with provider override**

```bash
curl -X POST http://localhost:3000/api/v1/extract \
  -H "Content-Type: application/json" \
  -d '{
    "content": "<html><body><p>Thanks for the update!</p><div class=\"gmail_quote\">...</div></body></html>",
    "mode": "llm",
    "provider": "anthropic"
  }'
```

**Example: .eml file upload**

```bash
# Read .eml file and pass as string
EML_CONTENT=$(cat email.eml)
curl -X POST http://localhost:3000/api/v1/extract \
  -H "Content-Type: application/json" \
  -d "{
    \"content\": $(echo "$EML_CONTENT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),
    \"is_eml\": true,
    \"mode\": \"auto\"
  }"
```

**Example response**

```json
{
  "success": true,
  "data": {
    "latest_message": "Hi, can we reschedule the meeting?",
    "confidence": 0.91,
    "flow_used": "non_llm (auto mode)",
    "metadata": {
      "original_length": 74,
      "after_reply_removal_length": 38,
      "after_signature_removal_length": 38,
      "after_disclaimer_removal_length": 38,
      "confidence_score": 0.91,
      "is_reliable": true,
      "reduction_ratio": 0.486
    }
  },
  "error": null,
  "agent_trace": null
}
```

---

### POST /api/v1/extract/stream

Stream extraction results as Server-Sent Events (SSE). Useful for displaying real-time LLM token output in the UI.

**Request Body**: same as `POST /api/v1/extract`

**Response**: `text/event-stream`

Each event is a line in the format:

```
data: <JSON payload>\n\n
```

#### SSE Event Types

| `type` | Additional fields | When emitted |
|--------|-------------------|--------------|
| `start` | `agent: string` | An agent or stage begins executing |
| `token` | `text: string` | A chunk of LLM output (streaming tokens, LLM mode only) |
| `agent_done` | `agent: string` | An agent finished |
| `done` | `result: object` | Extraction complete; `result` contains `latest_message`, `confidence`, `flow_used` |
| `error` | `message: string` | An error occurred |

#### Event sequence for `mode: "llm"` (streaming)

```
data: {"type": "start", "agent": "extraction"}

data: {"type": "token", "text": "Hi, can we"}
data: {"type": "token", "text": " reschedule..."}

data: {"type": "agent_done", "agent": "extraction"}

data: {"type": "start", "agent": "cleanup"}
data: {"type": "agent_done", "agent": "cleanup"}

data: {"type": "start", "agent": "confidence"}
data: {"type": "agent_done", "agent": "confidence"}

data: {"type": "done", "result": {"latest_message": "...", "confidence": 0.92, "flow_used": "llm_stream"}}
```

#### Event sequence for `mode: "non_llm"` (streaming)

```
data: {"type": "start", "agent": "non_llm"}
data: {"type": "token", "text": "<full extracted text>"}
data: {"type": "done", "result": {"latest_message": "...", "confidence": 0.88, "flow_used": "non_llm"}}
```

**Note**: The streaming endpoint does not run the full LangGraph validation loop. It runs extraction, cleanup, and confidence sequentially without the validation re-loop. For full agent fidelity, use the non-streaming `POST /extract` endpoint.

**Example**

```bash
curl -N -X POST http://localhost:3000/api/v1/extract/stream \
  -H "Content-Type: application/json" \
  -d '{"content": "Please confirm receipt.\n\nOn Mon wrote:\n> Will do.", "mode": "llm"}'
```

---

### GET /api/v1/history

Return extraction history (newest first, up to 1000 entries).

**Response** `200 OK` — array of `HistoryEntry` objects

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Short UUID (8 chars) |
| `timestamp` | string | ISO 8601 UTC timestamp |
| `mode` | string | Requested extraction mode |
| `flow_used` | string | Actual flow that ran |
| `confidence` | float | Confidence score |
| `latest_message` | string | Extracted message |
| `content_preview` | string | First 120 chars of original input (newlines replaced with spaces) |
| `provider` | string \| null | Provider override if specified |

**Example**

```bash
curl http://localhost:3000/api/v1/history
```

---

### DELETE /api/v1/history

Delete all extraction history entries.

**Response** `204 No Content`

**Example**

```bash
curl -X DELETE http://localhost:3000/api/v1/history
```

---

### GET /api/v1/providers

List all LLM providers and their configuration status.

**Response** `200 OK`

```json
{
  "providers": [
    {"name": "openai", "available": true, "configured": true},
    {"name": "azure_openai", "available": false, "configured": false},
    {"name": "gemini", "available": false, "configured": false},
    {"name": "anthropic", "available": false, "configured": false}
  ],
  "default_provider": "openai"
}
```

A provider is `available` and `configured` when its required API key (and for Azure: endpoint and deployment) are present in the environment. `default_provider` is the first available provider, or `"none"` if none are configured.

**Example**

```bash
curl http://localhost:3000/api/v1/providers
```
