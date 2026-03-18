# Spec: OpenClaw-compatible tool schema + tool-call normalization layer

Date: 2026-03-08

## Context
We want local models (HF / GGUF / EXL2 / vLLM/OpenAI) to be usable in OpenClaw-style agent loops.

OpenClaw’s current “tools” approach is:
- typed tools (no shelling as the primary interface)
- policy-driven allow/deny + profiles + provider/model overrides
- tools are presented to the model in **two parallel channels**:
  1) system prompt text (human-readable list + guidance)
  2) structured tool schema sent to the model API

Key implication (worth repeating because it drives the whole design):
- If a tool doesn’t appear in the system prompt text or the tool schema, the model cannot call it.

This spec defines a repo-local layer that:
- loads a tool schema (OpenAI-compatible function definitions)
- applies a policy (allow/deny, optional profiles/groups)
- sends tool schema to backends that support it (OpenAI-compatible servers like vLLM)
- provides a **tool-call normalization** pipeline so model-emitted tool calls can be converted into a canonical structure (even when emitted as text)

## Goals
- Make tool calling “plug-and-play” from this repo by standardizing:
  - tool schema inputs
  - tool-call outputs (canonical form)
  - streaming behavior (buffering partial tool calls)
- Prefer structured tool calls when the backend provides them; fall back to text parsing when it does not.
- Keep execution out of scope for v1: we surface tool calls reliably; OpenClaw (or later code) executes them.

## Non-goals (v1)
- Executing tools from the TUI.
- Perfect compatibility with every model’s proprietary tool format.
- Enforcing OpenClaw’s full policy language (we’ll start with a compatible subset).

## Definitions

### Canonical tool schema (input)
Use OpenAI-style “function tool” schema:
- `tools`: list of `{ type: "function", function: { name, description?, parameters } }`
- `tool_choice`: `"auto" | "none" | {type:"function", function:{name:"..."}}` (optional)

### Canonical tool call (output)
Internal representation (language-agnostic):
- `id`: optional string (server-provided if available)
- `name`: function name (string, required)
- `arguments_json`: JSON string (required; may be `{}`), not a dict to preserve exact bytes
- `status`: `partial | complete`
- `source`: `structured | parsed_text`

### “Structured-first” rule
If a backend provides tool calls as structured fields, they are the source of truth.
Do NOT also parse `content` for tool calls in that case (avoid double-processing).

## Config (TOML)
Add a new section that is backend-agnostic:

```toml
[tools]
enabled = false
schema_file = ""          # path to a JSON file (OpenAI tool schema)
policy_profile = "full"   # "minimal" | "coding" | "messaging" | "full" (subset)
allow = []                # optional allowlist, case-insensitive, supports group:* entries
deny = []                 # optional denylist, deny wins
parse_mode = "strict"     # "off" | "strict" | "lenient"

[tools.byBackend]
openai = { enabled = true }
hf = { enabled = false }
gguf = { enabled = false }
exl2 = { enabled = false }
ollama = { enabled = false }
```

Notes:
- `schema_file` is a repo asset (committed) unless it contains secrets.
- `byBackend` is a local analogue of OpenClaw provider/model restriction (subset).

## Tool policy (subset aligned with OpenClaw concepts)
OpenClaw supports `tools.allow` / `tools.deny` with wildcards and groups, and tool profiles that set a base allowlist.

Repo v1 policy subset:
1) Start with `policy_profile` base allowlist:
   - `minimal`: `session_status` only (if present in schema), else none
   - `coding`: allow `group:fs`, `group:runtime`, `group:sessions`, `group:memory`, `image` (when present)
   - `messaging`: allow `group:messaging`, `sessions_list`, `sessions_history`, `sessions_send`, `session_status` (when present)
   - `full`: allow everything in schema
2) Apply `allow` if non-empty (restrict to allowlist)
3) Apply `deny` (deny wins)
4) Expand `group:*` entries using a local mapping table (optional)

We do NOT implement `byProvider` in v1 (that is OpenClaw-side), but we support `byBackend`.

## Backend behavior

### `openai` backend (vLLM first)
- Send tool schema via the API’s `tools` field.
- Send `tool_choice` when configured (optional).
- Parse streaming `tool_calls` deltas (buffer per index) per `docs/specs/0010-openai-compatible-backend-vllm.md`.

### `hf` / `gguf` / `exl2` / `ollama` backends
v1: do not send structured tool schema (unless the engine provides an equivalent).

Instead, when `tools.enabled` is true:
- Inject a compact human-readable tool list into the system prompt text (system channel).
- Expect tool calls to be emitted as text, then normalize via parsing (below).

This matches OpenClaw’s “system prompt text” channel even when tool schema can’t be sent.

## OpenClaw tool shape note (important for schema authors)
OpenClaw tools are often “multi-action” tools:
- Example: a single `browser` tool handles actions like `status`, `snapshot`, `act`, etc via an `action`/`kind`-style argument.
- This means the OpenAI tool “function name” can be stable (`"browser"`), and the “subcommand” lives in `arguments`.

When building a schema file intended to mimic OpenClaw, prefer this pattern (single tool name with an enum field) over inventing separate tool names like `browser_snapshot`.

## Tool-call parsing + normalization

### Parsing sources
We support multiple parsers; each produces canonical tool calls.

Order of precedence (v1):
1) Structured tool calls from backend (OpenAI `tool_calls`)
2) Model-specific “tag protocols” (configured per model/backend)
3) JSON-in-text fallback (strict)

### Parsers (v1 set)
1) `openai_structured`:
   - Input: streaming delta fragments and/or final structured tool_calls array
   - Output: canonical tool calls
   - MUST buffer fragments per index; do not leak partial arguments into answer stream

2) `qwen_xmlish`:
   - Recognize Qwen-style `<tool_call>...<function=NAME>...<parameter=...>...</parameter>...</function>...</tool_call>`
   - Convert `<parameter=foo>bar</parameter>` into JSON arguments `{ "foo": "bar" }` (strings by default)
   - `strict` mode: reject malformed blocks
   - `lenient` mode: best-effort with explicit “uncertain” marker in meta

3) `json_object`:
   - Recognize a single top-level JSON object that looks like a tool call:
     - `{ "tool_call": { "name": "...", "arguments": ... } }` or
     - `{ "name": "...", "arguments": ... }`
   - `strict` mode requires valid JSON and a known tool name.

### Where parsing happens
- Streaming: buffer until a complete call is detected; only then emit a tool-call event.
- Non-streaming: parse the full text at end.

### How to display in the TUI (v1)
Keep it minimal:
- Append a non-executed tool call to transcript as an InfoMessage block:
  - tool name
  - arguments JSON (pretty-printed)
- Also store the canonical calls under `TurnRecord.gen["tool_calls"]`.

Do NOT attempt to execute the tool.

## History injection (tool results)
When integrating with OpenClaw (or later local execution), tool results should be injected as `role="tool"` messages.

This repo should preserve backends’ conventions:
- OpenAI: `role="tool"` messages.
- HF template-based: map to the model’s expected tool-response format if a parser/template requires it.

## Testing checklist
- A model that returns structured `tool_calls` (OpenAI backend):
  - tool call arguments stream correctly (buffered)
  - answer stream remains clean
- A Qwen-style model (tag protocol):
  - emits `<tool_call>` blocks; parser extracts name/args
- A “JSON prints in content” model:
  - strict JSON-only mode either parses cleanly or refuses with clear message
