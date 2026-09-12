# 0022 — OpenClaw / OpenAI Compatibility Optimization Pass

## Summary
Tighten our OpenAI-compatible behavior and diagnostics so OpenClaw (and other OpenAI-style clients) can reliably drive a tool-capable local model server (primarily vLLM) without edge-case protocol breakage.

This is an “optimization pass” focused on correctness, robustness, and debuggability—not adding new agent features.

## Background / Motivation
We already run tool-capable sessions via vLLM’s OpenAI-compatible server and our `openai_http` transport.
We also maintain a checklist for OpenClaw interoperability at `docs/openclaw/04-model-runner-openclaw-compat-checklist.md`.

Recent tool testing exposed that “mostly compatible” isn’t enough: small protocol mismatches (tool-call shapes, SSE edge cases, error frames, finish semantics) can cause confusing failures or loops.

## Goals
- Make the OpenAI-compatible path resilient to common API variants:
  - SSE streaming variations (multi-line frames, event fields, error frames).
  - Tool-call completion semantics (`finish_reason="tool_calls"`, empty content, etc.).
  - Tool-choice input variants (string + object forms) when used by upstream clients.
- Improve visibility:
  - Ensure request/response capture can show the exact payload that was sent and any error body/frames received.
  - Surface “sent/deferred/ignored” knob reporting consistently.
- Update the OpenClaw checklist to reflect the actual compatibility contract we want to hold.

## Non-goals
- Implementing full OpenAI Responses API orchestration semantics.
- Adding new tools beyond what’s already planned (see `0021-safe-browser-tool-v1.md`).
- Building a separate “OpenAI server adapter” for non-vLLM engines in this pass.

## Scope
This spec covers:
- `tui_app/transports/openai_http.py` (OpenAI-compatible HTTP + SSE parsing, tool-call deltas, request capture)
- `tui_app/backends/vllm.py` (managed vLLM launch ergonomics where it affects protocol behavior)
- `docs/openclaw/04-model-runner-openclaw-compat-checklist.md` (checklist improvements)

## Requirements

### R1. SSE parsing robustness
Our OpenAI-compatible stream parser MUST:
- Handle SSE frames that include fields other than `data:` (ignore unknown fields).
- Handle multi-line `data:` blocks (accumulate until blank line delimiter).
- Treat `[DONE]` as end-of-stream.
- Detect and surface server-side error frames:
  - `{"error": {...}}` at top-level
  - and/or error events that embed error text in a non-standard envelope
- On timeout or disconnect, emit a single clear error that includes:
  - backend name
  - url
  - elapsed
  - and the last captured error/body snippet when available

### R2. Tool-call response handling correctness
When streaming tool calls, we MUST support incremental accumulation of:
- `choices[0].delta.tool_calls[*].index`
- `...id`
- `...type`
- `...function.name`
- `...function.arguments` (string fragments)

We MUST correctly handle cases where:
- tool calls are present but `content` is `null`, empty, or whitespace
- finish is signaled with `finish_reason="tool_calls"`

### R3. Non-stream response support (compat)
Even if our TUI continues to use `stream=true` by default, the OpenAI HTTP transport SHOULD support `stream=false` responses for:
- test harnesses
- upstream compatibility checks

Minimum required:
- Parse `choices[0].message.content`
- Parse `choices[0].message.tool_calls` when present
- Capture `usage` when present

### R4. Tool-choice input compatibility
If a client/session specifies tool choice, we MUST accept and forward (when backend supports it) the OpenAI-compatible shapes:
- `"auto"`, `"none"`
- object form that selects a named tool/function

We MUST clearly report in `/show gen`:
- what was sent
- what was ignored (unsupported backend)

### R5. Tool role message tolerance
We MUST accept tool role messages that include optional fields used by some clients/servers:
- `name` on `role="tool"` messages (ignore if unsupported; preserve if harmless)

### R6. Multimodal content tolerance (documented)
We MUST make an explicit decision and document it (either):
- Support OpenAI-style content parts arrays for images on OpenAI-compatible backends, OR
- Explicitly mark it as unsupported for OpenClaw integration and fail with a clear error

### R7. Checklist alignment
`docs/openclaw/04-model-runner-openclaw-compat-checklist.md` MUST be updated to include:
- `finish_reason` expectations
- `tool_choice` accepted forms
- SSE error frame behavior
- tool message field tolerance (`name`)
- `stream=false` tool_calls location (`choices[0].message.tool_calls`) (for completeness)
- explicit multimodal stance

## Proposed implementation notes (high level)
- Extend the SSE parser to true SSE semantics (blank-line delimited events) rather than “one JSON per `data:` line”.
- Ensure request capture stores:
  - request payload (sanitized)
  - response error bodies/frames (sanitized, truncated)
- Add config/CLI key(s) for tool choice if not already present (or explicitly defer if not needed yet).
- Keep all compatibility behavior backend-gated:
  - don’t assume vLLM-only semantics for external OpenAI-compatible servers

## Acceptance criteria
- A streamed tool-call turn works end-to-end with:
  - tool call deltas arriving fragmented
  - `finish_reason="tool_calls"`
  - assistant `content` empty/`null`
  - follow-up tool messages with `tool_call_id`
  - subsequent assistant completion
- Any server-side tool/template error is surfaced to the user with:
  - clear context (“server error frame”, url)
  - and is visible in logs/request capture
- Updated checklist is consistent with the implemented behavior.

## Test plan
- Add a small “compat smoke” script (local-only) that:
  - calls `/v1/chat/completions` with `stream=false` and `stream=true`
  - includes a minimal tool schema and validates:
    - tool_calls shape in stream deltas
    - tool_calls shape in non-stream `message.tool_calls`
    - finish_reason values
    - usage parsing when present
- Run the same smoke against:
  - managed vLLM
  - an external OpenAI-compatible server (when available)

