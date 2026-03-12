# Spec: OpenAI-compatible backend (vLLM) + “backend build” workflow

Date: 2026-03-07

## Context
We now support TOML config layering (base + profile + machine overrides) per `docs/specs/0009-toml-config-profiles-and-machine-overrides.md`.

Next goal: run local models in a way that is compatible with OpenClaw-style agent loops and tool calling.
vLLM is a strong candidate because it can expose an **OpenAI-compatible** HTTP API with streaming.

This spec covers two parts:
1) **Backend build**: how to run a vLLM server for a chosen HF-format model (local path).
2) **Client integration**: how this repo’s unified TUI should talk to an OpenAI-compatible server backend.

## Goals
- Add a new backend: `openai` (OpenAI-compatible HTTP server).
- First supported engine: **vLLM** (others can follow later using the same backend).
- Support streaming tokens into existing TUI events (`ThinkDelta` / `AnswerDelta`).
- Support OpenAI-style tool calls *as raw text/events* (no execution yet).
- Support TUI image attachments for OpenAI backends using data URLs (best-effort).
- Keep dependencies optional: vLLM is not required unless `--backend openai` is used.

## Non-goals (v1)
- Running GGUF/EXL2 models through vLLM (not supported by vLLM; these remain separate engines).
- Full tool execution loop UI (OpenClaw execution). We only surface tool-call content.
- Perfect parity of all generation knobs across engines.

## Terminology
- **Model format**: HF folder (Transformers), GGUF, EXL2, Ollama model name.
- **Engine**: Transformers, llama.cpp, ExLlamaV2, Ollama server, vLLM server.
- **Backend** (repo): the adapter that produces TUI events (HF/GGUF/Ollama/EXL2/OpenAI-compatible).

## 1) vLLM “backend build” (server side)

### Recommended build strategy
- Run vLLM as a separate long-running process, then point this repo at it.
- Put server endpoint defaults in `config/machine.toml` (gitignored).

### Minimal server start (text-only)
Example (exact flags may vary by vLLM version; keep it simple):
```bash
vllm serve /home/poop/ml/models/Qwen3.5-9B \\
  --host 127.0.0.1 --port 8000
```

### Vision / multimodal
If the model is vision-capable, enable whatever vLLM requires for multimodal inputs for your installed version.
Goal is to accept OpenAI-like message content segments that include image URLs / data URLs.

### Tool calling
vLLM’s OpenAI server supports OpenAI-compatible tool fields at the HTTP layer.
Whether the model actually produces correct tool calls depends on:
- the checkpoint (instruct vs base)
- prompting/tool schema
- decoding settings

### Debug endpoints (sanity)
- `GET /v1/models` should list the served model name(s).
- A basic streaming chat completion request should stream token deltas.

## 2) Repo integration: `openai` backend

### CLI / detection
Add:
- `--backend openai`
- Support a model ref scheme: `openai:<model_name>` (optional)

Default backend selection remains unchanged; `openai` must be explicit in v1 to avoid ambiguity.

### Config (TOML)
Add new template:
- `models/_TEMPLATE/openai/config/default.toml`

Shape:
```toml
[model]
id = ""               # model name as known by the server (required; see “Model resolution”)

[gen]
stream = true
max_new_tokens = 1024
temperature = 0.7
top_p = 1.0
stop_strings = []

[prompt]
system = ""
system_file = ""
user_prefix = ""

[ui]
show_thinking = true
assume_think = false

[backend.openai]
base_url = ""         # e.g. http://127.0.0.1:8000/v1
api_key = ""          # optional; prefer env var if used
timeout_s = 600       # total request timeout (see “Timeout semantics”)
```

### Machine overrides
Extend `config/machine.toml` support:
- `openai_base_url` should fill `[backend.openai].base_url` if it’s unset.

Important: machine overrides must not rewrite HF Hub ids into local paths.
If `model.id` looks like `org/name`, treat it as a server-side model name, not a filesystem path.

### Model resolution (MUST)
In practice, many vLLM/OpenAI-compatible servers require the `model` field.

v1 behavior:
- If `[model].id` is set (non-empty), the backend MUST send it as the request `model`.
- If `[model].id` is empty:
  1) the backend MUST call `GET /v1/models` once at startup
  2) if exactly one model is returned, the backend MUST use that model id
  3) otherwise, the backend MUST fail with a clear error listing available model ids and instructing the user to set `[model].id`

### API contract (v1 MUST target)
v1 targets the OpenAI Chat Completions API shape:
- Request: `POST /v1/chat/completions`
- Streaming: Server-Sent Events (SSE)
  - Frames are lines beginning with `data: `
  - End marker: `data: [DONE]`
  - Error frames must be surfaced as a single `Error(...)` event with the server-provided message

Streaming JSON contract (common subset):
- `choices[0].delta.content` (text chunks)
- `choices[0].finish_reason` (terminal reason)
- `choices[0].delta.tool_calls[]` (tool-call fragments; see below)

### Payload building contract (v1)
The backend MUST build an OpenAI Chat Completions request with:
- `model` (string): resolved per “Model resolution”
- `messages`: list of `{role, content}` objects
  - `content` may be:
    - string (text-only)
    - list of segments (multimodal):
      - `{ "type": "text", "text": "..." }`
      - `{ "type": "image_url", "image_url": { "url": "data:image/png;base64,..." } }`
- generation knobs mapping:
  - `max_new_tokens` MUST be sent as `max_tokens` (v1 requirement)
  - `temperature`, `top_p`
  - `stop_strings` MUST be sent as `stop` when present
  - `seed` only if server supports it (otherwise treat as deferred)

The backend MUST report (once at startup and via `/show gen`) which knobs are:
- **sent** (included in the HTTP payload)
- **deferred** (not sent; server/model defaults apply)
- **ignored** (configured but the backend does not implement them)

### Streaming event mapping
For OpenAI-style streaming responses, map server deltas to TUI events:
- If the server yields a dedicated reasoning field (server-specific; e.g. `delta.reasoning` or `delta.reasoning_content`), route that to `ThinkDelta`.
- Else: run `ThinkRouter` on streamed text to split `<think>...</think>` markers.

Stop conditions:
- Respect server termination and emit `Finish` with a `TurnRecord` containing:
  - `raw`: concatenated streamed content (and tool call text if included)
  - `think`/`answer`: extracted via router if applicable
  - `gen`: snapshot of “sent” payload values (not “configured but omitted”)

### Tool-call streaming contract (v1 MUST)
Tool calls may stream incrementally under `choices[0].delta.tool_calls`.

v1 MUST:
- Buffer tool call fragments per tool index (and per tool id if provided), accumulating:
  - function name
  - arguments string fragments
- Do NOT interleave partial tool-call JSON/arguments fragments into `AnswerDelta`.
- On `Finish`, append a stable textual representation to `TurnRecord.raw` (and optionally store parsed tool calls under `TurnRecord.gen["tool_calls"]`).

Example stable text representation appended to `raw`:
```
<tool_call>
<function=name>
<arguments>
{...}
</arguments>
</function>
</tool_call>
```

### Image attachments from the TUI (`/image`) (v1 MUST)
`/image` currently exists in the TUI.

v1 MUST update behavior:
- Allow `/image` when backend is `hf` OR `openai`.
- Deny (with a clear message) for `gguf`, `ollama`, `exl2`.

For `openai` backend:
- Convert local image file paths to `data:` URLs (base64) at request time.
- Enforce local size caps (fail-fast):
  - **10 MB per image** (raw file size on disk)
  - **20 MB total** across all attached images for the turn
- Error message MUST suggest remediation:
  - resize the image
  - attach fewer images
  - or use a hosted URL (future)

For `hf` backend:
- Keep current local-path behavior (processor loads images from disk).

### Auth precedence and secrecy (v1 MUST)
Auth is optional for local vLLM but should be supported.

Precedence for API key:
1. CLI `--openai-api-key` (if provided)
2. Environment variable `OPENAI_API_KEY`
3. Config `[backend.openai].api_key`
4. Machine config `config/machine.toml` (if we ever add it; otherwise skip)

Rules:
- Never print the API key value in `/show *`, exceptions, or logs.
- `/show backend` MAY only show `api_key: (set)` / `(unset)` and the source label, never the value.

### Timeout semantics (v1 MUST)
`timeout_s` MUST be treated as a **total request timeout** (connect + read) for the chat completion request.
If the HTTP client supports separate connect/read timeouts, v1 uses:
- connect timeout = `min(10, timeout_s)`
- read timeout = `timeout_s`

## Testing checklist
- Config resolution:
  - base TOML only
  - base + profile
  - base + machine override
  - CLI overrides above all
- Server connectivity errors are surfaced clearly.
- Streaming works:
  - tokens stream without UI stutter
  - thinking routing works (when model emits think tags)
- Image path attach:
  - `/image` + prompt produces a multimodal request for openai backend
  - large image fails fast with a clear message

## Rollout plan
1. Add `openai` backend skeleton + config template.
2. Add machine override mapping for `openai_base_url`.
3. Add a minimal streaming client implementation (no tools/images first).
4. Add tool-call surfacing (no execution).
5. Add image segment serialization (data URL) as an optional enhancement.
