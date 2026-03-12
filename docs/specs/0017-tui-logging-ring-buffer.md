#!/usr/bin/env markdown
# Spec: TUI logging ring buffer (backend-standard)

Date: 2026-03-09

Status: Draft

Related audit:
- `docs/audits/0004_tui_logging.md`

Related umbrella contract:
- `docs/specs/0015-backend-standardization-contracts.md`

## Context / problem
We want `/show logs` to work consistently across **all** TUI backends, regardless of whether file logging is enabled.

Today we have backend-specific logging patterns (some use file-only logging; some log only to stdout; some can surface a small tail; some can’t).
This makes in-TUI debugging inconsistent and forces “tee to file” even for simple investigations.

## Goals
- Every backend session maintains an **in-memory ring buffer** of recent log lines.
- `/show logs` works for every backend (at least tail + metadata).
- Timestamps are explicit and unambiguous:
  - UTC
  - ISO 8601: `YYYY-MM-DDTHH:MM:SS.sssZ`
- Optional file sink stays supported, but is not required for `/show logs`.
- Redaction rules are standardized (never leak API keys / auth headers / data URLs).
- Allow richer backends (managed engines) to capture multiple sources (stdout/stderr) without forcing that complexity onto every backend.

## Non-goals (v1)
- A full structured logging system (levels, JSON logs everywhere).
- “Live tail” that updates without a command (we can add later).
- Guaranteed capture of stdout/stderr from third-party processes (handled by managed engines as best-effort).

## Standard interface (required)
All backend sessions MUST implement:
- `get_recent_logs(n: int = 80, sources: list[str] | None = None) -> list[str]`

Behavior:
- Returns up to `n` most recent log lines, newest last.
- Lines are already formatted with timestamp and source prefix (format invariants below).
- If backend has no logger configured, it still returns ring-buffer contents.
- If ring buffer is empty, returns `[]`.

All backend sessions SHOULD implement:
- `list_log_sources() -> list[str]`

This enables `/show logs` to show what sources exist in the current session.

## Data model / formatting rules
Each stored line MUST have:
- `ts_utc`: `YYYY-MM-DDTHH:MM:SS.sssZ`
- `source`: short stable source tag (see Source tag contract below)
- `message`: single-line text (newline characters replaced with `\\n`)

Recommended line format:
- `<ts_utc> [<source>] <message>`

Important: although the logical record has fields (`ts_utc`, `source`, `message`), the public interface returns `list[str]`.
These fields are therefore **formatting invariants of the rendered string**, not a structured return type.

## Source tag contract (MUST)
To avoid per-backend drift, sources MUST use a small standard vocabulary when possible.

Minimum required source tags:
- `app` (TUI / session-level orchestration)
- `backend` (backend engine wrapper: hf/gguf/exl2/openai_http/etc.)
- `transport` (HTTP streaming, request/retry plumbing, etc.)

Managed-engine backends (e.g. vLLM managed mode) SHOULD also provide:
- `engine_stdout`
- `engine_stderr`

Notes:
- Backends MAY add additional sources, but SHOULD prefer the standard tags above.
- `/show logs --filter` is a TUI feature; the logger stays simple and only supports optional source filtering at retrieval time.

## Redaction requirements (MUST)
Before a line enters the ring buffer (and before writing to file), apply the following pipeline (MUST, in order):
1. Normalize to a single line (replace newlines with `\\n`).
2. Redact secrets.
3. Truncate to hard limits.

Redaction rules (MUST):
- API keys: replace any known API key fields with `***`.
- Authorization headers: replace with `***`.
- Image data URLs: replace with a placeholder like `<image:data_url omitted size=<n>>`.
- URLs with embedded secrets (bearer tokens in query strings, `token=` params, etc.) MUST be redacted.
- Extremely long payloads: truncate (see hard limits below).

Hard limits (MUST):
- Per-line max: 8 KiB (truncate with `…<truncated>` marker).
- Ring buffer total: configurable (default 500 lines).

## File sink (optional)
If configured, a backend MAY write the same formatted lines to a file sink.

Requirements:
- File sink does not replace ring buffer; it mirrors it.
- File path resolution MUST be explicit and handled by the shared logger helper:
  - if path is relative, resolve relative to `args._config_path` directory when available; otherwise resolve relative to CWD.
- If file sink fails:
  - session continues (log to ring buffer only)
  - emit a one-time warning line into the ring buffer (MUST be one-time; MUST not spam).

## TUI command behavior
`/show logs`:
- Shows the tail from `get_recent_logs()`.
- If `list_log_sources()` exists, show the available sources (once) before the tail.
- If empty: prints `No logs captured for this session.`

## Implementation sketch
1. Add a shared helper (module) that implements a session-owned logger object:
   - ring buffer storage (default 500 lines, configurable)
   - formatting + UTC timestamping (`YYYY-MM-DDTHH:MM:SS.sssZ`)
   - redaction + truncation
   - optional mirrored file sink with config-relative path resolution
   - `warn_once(key, message)` helper for sink failure and similar one-time warnings
2. Replace backend-specific “log tail” ad-hoc code with the shared logger.
3. For managed engines (e.g. vLLM):
   - continue capturing child process output if already available
   - route stdout/stderr into the shared session logger with `engine_stdout` / `engine_stderr` sources
   - do not downgrade richer backends to match simpler ones

## Compatibility / migration guidance
- vLLM’s richer logging approach can remain; the only requirement is that it feeds into the shared session logger so `/show logs` is uniform.
- Backends that currently only have file logging can migrate by:
  - replacing direct file writes with `session_logger.log(source=..., message=...)`
  - optionally enabling the file sink in the logger.

## Testing checklist
- Start each backend with no log file configured:
  - `/show logs` still returns useful lines.
- With a log file configured:
  - `/show logs` matches file tail content (within truncation limits).
- Confirm redaction:
  - `/show args` never prints API keys
  - `/show request --verbose` never prints image data URLs
  - `/show logs` never prints secrets
