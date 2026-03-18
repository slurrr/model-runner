# 0004 TUI Logging

Date: 2026-03-08

## Purpose
Reference for standardizing backend logging surfaced in the TUI (`/show logs`) so diagnostics are consistent across backends.

This is a supporting audit for backend standardization in `docs/audits/0003-standardization.md`.

## Current State

### Shared TUI surface
- All backends now expose `get_recent_logs(n)` so `/show logs` works everywhere.
- For non-vLLM backends, log lines are produced by app-level logging calls (not raw engine stdout/stderr).

### vLLM (managed)
- Captures child-process `stdout` and `stderr` into an in-memory deque.
- Optional tee-to-file supported.
- This gives highest-fidelity diagnostics in-TUI today.

### HF / GGUF / EXL2 / Ollama / OpenAI
- Use `FileLogger` ring buffer (in-memory, optional file sink).
- Captures structured app-level lifecycle/events/errors we emit.
- Does **not** automatically capture raw library/framework stderr/stdout internals.

## Target Standard
Adopt one backend log buffer contract with source tagging and worker/session ownership:
- `append_log(source, level, message, context?)`
- `get_recent_logs(n, filter?)`
- Optional `get_log_file_path()`

Recommended `source` values:
- `app` (our runner/session code)
- `engine_stdout`
- `engine_stderr`
- `transport`

This keeps one retrieval path in TUI while allowing backend-specific capture methods.

Hard requirements:
- Ownership: log buffers are per worker/session, never global.
- Timestamp: UTC, ISO 8601 with milliseconds: `YYYY-MM-DDTHH:MM:SS.sssZ` (avoid local timezone ambiguity).
- Safety: never log secrets; redact auth headers/api keys; omit or summarize data URLs (images).

## Standard Runtime Modes

### 1) Always-on in-memory (fast path)
- Ring buffer per backend worker/session.
- Keep small fixed size (recommended: 200-500 lines).
- Store compact structured lines only:
  - `timestamp source level message` (timestamp is UTC `...Z`)
- No file writes by default.
- `/show logs` reads this buffer.

### 2) Optional file sink (debug mode)
- Off by default.
- Enable per backend/model only when needed.
- Default file level should be `warn/error` unless debugging requires more.
- Minimal retention:
  - single rotation
  - small cap (recommended: 1-2 MB + 1 backup)

Default posture:
- in-memory on
- file off
- enable file only for stubborn failures (especially pre-init crashes).

## Worker-First Parity Plan
To match vLLM-level diagnostics across backends, prefer managed backend workers where possible:
- capture worker `stdout/stderr`
- feed captured lines into the same in-memory buffer API
- optionally mirror selected lines to debug file sink

For in-process backends that cannot be isolated immediately:
- keep structured app-stage logs
- add stage coverage (init/build request/generate/retry/failure)
- treat worker isolation as the long-term parity target

## Performance Guidance
- Avoid verbose logging in hot token loops.
- Log stage transitions and failures, not every token/chunk.
- Downsample repetitive backend spam before appending to buffer.
- Use non-blocking append; drop oldest entries when full.
- Logging must never block generation.

## Practical Conclusion
- Near-term: keep `/show logs` consistent with compact in-memory buffers everywhere.
- Debug/deep-failure path: turn on minimal file sink as needed.
- Long-term parity: move backends toward worker-managed execution so TUI can surface raw engine startup/runtime logs uniformly.
