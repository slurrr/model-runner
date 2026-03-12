#!/usr/bin/env markdown
# Spec: MVP tool execution harness in the TUI (safe, debuggable)

Date: 2026-03-11

Status: Draft

Related specs:
- `docs/specs/0011-openclaw-tool-compat-layer.md` (schema + normalization, no execution)
- `docs/specs/0015-backend-standardization-contracts.md` (contracts)
- `docs/specs/0017-tui-logging-ring-buffer.md` (logging contract)

## Context / problem
We want to test “tool-capable” models (e.g. Qwen) locally in this repo before using them in OpenClaw.

Today:
- vLLM/OpenAI transport can parse streamed `tool_calls`, but we do not execute them.
- HF/GGUF/EXL2 can be prompted to emit tool calls as text, but there is no shared loop.

This spec adds a minimal tool execution harness to the TUI, focused on:
- correctness and debuggability
- safety by default
- minimal initial tool surface (pure/safe tools)

## Goals
- Add an opt-in, local tool loop:
  - model emits tool call(s)
  - repo executes safe tools
  - tool results are fed back to the model
  - model produces final answer
- Provide excellent debugging:
  - tool call blocks in the transcript show the exact call shape + the result/error
  - tool activity is logged to the session logger (ring buffer + optional file sink)
- Minimize new “special prompting” requirements:
  - prefer structured tools when the backend supports it (vLLM/OpenAI)

## Non-goals (MVP)
- Network / filesystem / shell tools (dangerous surface area).
- OpenClaw’s full policy language and multi-provider tool routing.
- Parallel tool execution.
- Streaming UI for partial tool-call arguments (show complete calls only).

## MVP tools (built-in)
Start with 3 tools that are safe and deterministic:
1. `calc(expression: str) -> str`
2. `get_time(tz: str = "America/Denver") -> str`
3. `echo(text: str) -> str`

Notes:
- These are enough to validate that a model can select tools and produce valid arguments.
- Add more tools later behind explicit allowlists.

### `calc` safety requirements (MUST)
`calc` MUST NOT use Python `eval` (directly or indirectly).

Define a strict allowed arithmetic grammar, for example:
- numeric literals (ints/floats)
- parentheses
- operators: `+ - * / // % **`
- unary `+/-`

Implementation must:
- parse with `ast.parse(..., mode="eval")`
- reject any nodes outside an allowlist
- cap input length (e.g. 200 chars)
- cap exponent size / recursion depth to avoid DoS
- return a string result (or a structured error string)

## Config (TOML)
Add a backend-agnostic tools section (extends the shape introduced in spec 0011).

```toml
[tools]
enabled = false                # master switch
mode = "dry_run"               # "off" | "dry_run" | "execute"
schema_file = ""               # optional path to OpenAI tool schema JSON
allow = []                     # optional allowlist (names). If empty: allow all from schema/builtins
deny = []                      # deny wins
max_calls_per_turn = 3
timeout_s = 10
max_result_chars = 8000        # truncate tool output before feeding back / displaying
```

Behavior:
- `enabled=false` or `mode="off"` disables all tool handling (current behavior).
- `mode="dry_run"` displays tool calls but does not execute them (see dry_run contract below).
- `mode="execute"` executes tools and runs the follow-up model call(s).

## Tool schema sources
Tool definitions presented to the model come from:
1. `schema_file` if provided (OpenAI tools JSON, committed in-repo when possible)
2. built-in schema for the MVP tools (always available)

Merge rule (MUST):
- Final tool set exposed to the model is the union of (schema_file tools) ∪ (built-ins).
- Name collisions MUST be handled explicitly:
  - default: fail fast with a clear error listing the colliding names
  - optional future knob: allow schema_file to override built-ins (out of scope for MVP)

Policy application:
- If `allow` is non-empty, restrict to allowlisted tool names.
- Apply `deny` after allow (deny wins).

## Backend behavior (MVP scope)

### vLLM (managed) + OpenAI (attach)
Primary path for MVP.

Request:
- Send `tools=[...]` in the OpenAI Chat Completions payload when tools are enabled.
- `tool_choice` remains out of scope for MVP (defaults to auto).

Response parsing:
- Use structured `tool_calls` deltas (buffered per index) as source of truth.
- Do NOT parse `content` for tool calls when structured tool calls are present.
- If assistant content and tool_calls arrive together, preserve both.

Tool loop:
1. First model call produces an assistant message that MAY include both:
   - `content` (assistant text), and/or
   - `tool_calls` (one or more tool calls)
2. The client MUST preserve and re-send the assistant tool-call message as part of history:
   - append an assistant message with `role="assistant"`, `content`, and `tool_calls` (as received/assembled)
3. For each tool call (bounded by `max_calls_per_turn` across the whole outer turn):
   - validate name against allow/deny policy
   - execute tool (when mode=execute)
   - append `role="tool"` message with `tool_call_id` linkage and tool result content (or error content)
4. Call the model again with updated messages to produce the final answer (or additional tool calls).
5. Repeat until:
   - the model produces no new tool calls, or
   - `max_calls_per_turn` is reached, or
   - an unrecoverable error occurs.

### HF / GGUF / EXL2
Out of scope for MVP execution.

However, the transcript UI should still be able to display “tool call blocks” if the backend emits canonical tool calls in `TurnRecord`
(for later parity).

## Canonical message + record shape (MUST for MVP)
To avoid lossy history replay and enable tool-aware trimming, the tool harness MUST use a richer message shape internally.

Minimum required message fields (OpenAI-compatible shape):
- `role: "system"|"user"|"assistant"|"tool"`
- `content: str | None` (assistant content may be absent/null for tool-only assistant messages; do not invent content)
- `tool_calls: list[...]` (assistant-only; optional)
- `tool_call_id: str` (tool-only; required when responding to a tool call)

TurnRecord MUST preserve (tool activity):
- assembled tool call arguments string exactly as emitted/assembled (for debugging)
- parsed JSON arguments object when parsing succeeds (optional field)

Compatibility (MUST for MVP):
- Tool calls MAY be stored in a dedicated `TurnRecord.tool_activity` structure.
- Tool calls MUST also remain mirrored in `TurnRecord.gen["tool_calls"]` (existing behavior) for one phase, to avoid breaking `/show gen`
  and any downstream consumers. A later spec can migrate/soft-deprecate the `gen.tool_calls` mirror.

## Context trimming with tools (MUST)
History trimming MUST treat tool exchanges as an atomic unit.

Definition: a “tool exchange” is:
- 1 assistant message containing `tool_calls` (call ids set A)
- followed by all contiguous `role="tool"` messages whose `tool_call_id` is in A

Notes:
- This definition is unambiguous even when multiple tool rounds occur in one outer user turn, because each tool exchange is identified by
  the assistant message’s call id set A.

Rule:
- When trimming oldest history, drop tool exchanges as a unit (assistant tool_calls message + its tool result messages).
- Never drop the assistant tool_calls message while retaining its tool results, or vice versa.

## Transcript UI requirements
When a tool call happens, the transcript MUST show a distinct tool block that is easy to inspect:
- tool name
- tool_call_id (if present)
- arguments:
  - raw assembled arguments string (exact bytes as assembled from deltas)
  - parsed JSON pretty-print when parsing succeeds
- result or error (truncated to `max_result_chars`)

Minimum UX:
- show blocks after the call completes (no partial-argument streaming required)
- preserve the exact call bytes for debugging

## Logging requirements
All tool activity MUST go through the session logger (ring buffer + optional file sink):
- tool_call_received
- tool_call_execute_start
- tool_call_execute_success (include output length)
- tool_call_execute_error
- tool_call_result_injected

Redaction:
- never log secrets (rely on logger redaction pipeline)
- never log image data URLs (not in MVP, but keep invariant)

Source tags:
- Prefer `app` for orchestrator events.
- Prefer `transport` for HTTP request/response diagnostics.

## Persistence / transcript saving
If `--save-transcript` is enabled, the saved record MUST include tool activity for that turn.

Recommended representation:
- Store canonical tool calls and results on the `TurnRecord` (new fields), rather than only embedding into `raw` text.

## Error handling and safety
- If a tool call references an unknown tool name:
  - log error
  - surface an in-transcript tool error block
  - inject a tool result indicating the tool is unavailable (so the model can recover)
- Enforce:
  - `max_calls_per_turn`
  - `timeout_s`
  - `max_result_chars` truncation
- No implicit enabling of dangerous tools.

`max_result_chars` truncation (MUST):
- Apply truncation once at the tool executor boundary, before:
  - transcript display
  - tool-result injection back to the model
  - log persistence
- This keeps all three surfaces consistent and avoids “full result leaked in logs” surprises.

## /show / commands (MVP)
Add:
- `/show tools`:
  - show tools enabled/mode, schema source, allow/deny, and list tool names exposed to the model
  - in `--verbose`, include the schema path(s) and the post-policy filtered tool list

Alias:
- `/tools` is an alias for `/show tools`

## `dry_run` behavior (MUST)
`dry_run` exists to test “does the model produce tool_calls and are they shaped correctly” without executing anything.

Rules:
- In `dry_run`, the client MUST still send `tools=[...]` (otherwise you are not testing tool calling).
- If the model emits tool_calls:
  - show tool blocks in the transcript
  - DO NOT execute tools
  - DO NOT inject synthetic tool results
  - End the outer turn after the tool_calls are displayed (do not auto re-call), because re-calling without results often loops.
  - The turn may end with no final answer; use the existing “no final answer” UX path (no special one-off messaging).
- If the model emits no tool_calls, the turn behaves like normal chat.

Denied tool calls in `dry_run`:
- Still show the tool block with status `denied_by_policy`.
- Do not inject synthetic tool results (since we do not continue the loop in `dry_run`).

## Implementation plan (suggested)
1. Add a small `tui_app/tools/` package:
   - builtin tools + OpenAI schema representation
   - registry + executor with allow/deny policy
2. Extend the OpenAI transport to attach `tools=[...]` when enabled.
3. Add an orchestrator loop for OpenAI/vLLM sessions:
   - execute tool calls
   - inject `role="tool"` messages
   - re-call for final answer
4. Add transcript tool blocks + `/tools`.
5. Add saved transcript serialization for tool activity.

## Testing checklist
- With tools enabled (`mode=execute`), prompt the model to use `calc`:
  - tool call appears in transcript
  - result injected
  - model produces final answer referencing tool result
- With `mode=dry_run`:
  - tool call appears in transcript
  - no tool execution occurs
- With denylist:
  - model attempts a forbidden tool
  - tool error block appears and model recovers
