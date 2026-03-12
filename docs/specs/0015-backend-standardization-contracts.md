#!/usr/bin/env markdown
# Spec: Backend standardization contracts (TUI)

Date: 2026-03-08

## Context / problem
This repo grew from “a few runner scripts” into a multi-backend playground with a unified Textual TUI (`tui.py` + `tui_app/`).

We now have:
- a solid shared UI
- multiple engines/protocols with different capabilities

But we lack a single, explicit “contract” that every backend must meet, which makes it easy to:
- accidentally implement inconsistent UX
- silently accept placebo knobs
- lose debuggability when adding new backends/models

This spec defines the repo-wide backend contracts that keep the TUI consistent.

Supporting audits:
- `docs/audits/0003-standardization.md`
- `docs/audits/0004_tui_logging.md`

## Goals
- Make backend additions predictable:
  - consistent thinking behavior
  - consistent debugging surfaces (`/show`, logs, last request)
  - consistent metrics (token counts, tok/s) when available
- Avoid abstraction tax:
  - do not force full knob parity across engines
  - do not require heavyweight tokenizers purely for metrics

## Non-goals
- Refactor all backends in one PR.
- Add new UI features unrelated to consistency.

## Terms
- **Engine backend**: what runs the model (HF, GGUF, EXL2, vLLM managed, Ollama).
- **Transport**: how we talk to an out-of-process engine (OpenAI-compatible HTTP, Ollama HTTP).
- **Template control level**: whether the repo can enforce chat template/history shaping for a backend.

## Contract: Backend surface (must)
Every backend must implement the `BackendSession` protocol and follow the event contract:
- Emits `TurnStart(turn_id)` exactly once per turn.
- Emits any number of:
  - `ThinkDelta`
  - `AnswerDelta`
  - `Meta`
- Terminates with exactly one of:
  - `Finish(turn_id, record=TurnRecord)`
  - `Error(turn_id, message=...)`

Required session methods/fields:
- `backend_name: str`
- `resolved_model_id: str`
- `generate_turn(turn_id: int, messages: list[dict[str, object]], emit: EventEmitter) -> None`
- `get_recent_logs(n: int = 80) -> list[str]`

Recommended optional methods:
- `describe() -> dict[str, object]` (for `/status` and `/show backend`)
- `get_last_request() -> dict | None` (for `/show request` parity)

## Contract: Thinking behavior (must)
User-facing behavior:
- Thinking is routed to the thinking panel (`ThinkDelta`).
- Final output is routed to the answer panel (`AnswerDelta`).

Standard routing:
- For text-streaming backends, use `ThinkRouter(assume_think=args.assume_think)` to split think vs answer based on markers.
- If a backend provides a dedicated reasoning/thinking channel, emit that text via `ThinkDelta` directly, and route any remaining “content” via `ThinkRouter`.

Notes:
- Do not force “think tags” to exist; not all models emit them.
- `assume_think` exists specifically for models that emit only `</think>` (or emit markers inconsistently).

## Contract: Token accounting and metrics (must where available)
Tokens are the unit for:
- context windows
- generation budgets
- throughput measurements (tok/s)

Source of truth order (preferred):
1. Backend-native usage counters / token IDs
2. Backend tokenization API
3. Retokenization fallback (last resort, only if already loaded)

Minimum Meta keys (when knowable):
- `prompt_tokens: int`
- `completion_tokens: int`
- `total_tokens: int`

UI rules:
- `/show last` and `/status` must prefer token metrics when available.
- If token metrics are unavailable, show `unavailable` explicitly.
- Character counts may still be shown as a secondary diagnostic signal, but never presented as tokens.

See ADR:
- `docs/decisions/0015-token-accounting-and-metrics.md`

## Contract: Knob mapping semantics (must)
We will not standardize “every knob works everywhere”.
We will standardize **reporting** so users know what actually happened.

Per-turn (preferred) or per-session (minimum) reporting:
- `sent`: values that were actually applied/sent to the engine/server
- `deferred`: values left unset, meaning backend defaults apply
- `ignored`: values the user set but the backend cannot honor

The UI (`/show gen`) must surface this in a consistent format.

## Contract: Logging (must)
Use the standardized log buffer contract from:
- `docs/audits/0004_tui_logging.md`

Hard requirements:
- per-session/worker ring buffer (never global)
- UTC timestamp format: `YYYY-MM-DDTHH:MM:SS.sssZ`
- source tagging (at least: `app`, `engine_stdout`, `engine_stderr`, `transport`)
- redaction: never print secrets; omit or summarize data URLs
- logging must not block generation; do not log per-token

## Contract: Chat template and history control (best-effort)
Knob:
- Standardize a single `chat_template` config/CLI knob across backends where it is meaningful.

Capability:
- Every backend declares `template_control_level`:
  - `local_template`: template is rendered locally (HF/GGUF/EXL2)
  - `managed_server_template`: repo controls server launch template (vLLM managed)
  - `server_owned_template`: template is not realistically controllable by this repo (Ollama `/api/chat`, arbitrary external OpenAI servers)

Rules:
- If `server_owned_template`, `chat_template` must be reported as ignored (or “unsupported”) and must not claim to have taken effect.
- History sanitization rules (e.g., stripping `<think>...</think>` from the text we re-feed) apply to the **messages we send on future turns**,
  regardless of template control level.
  - This is a message-history policy, not a template-formatting policy.
  - It does not retroactively change what a model “already saw” in a completed turn; it only affects future requests.

## Contract: Context management (must)
Backends must converge on the same user-facing context-fit behavior even if the implementation mechanism differs.

Required user-visible effect:
- Drop oldest prior conversation history first when context must be reduced.
- Preserve the current user turn.
- Preserve the system message when possible.
- Reserve room for generation instead of filling the entire context with prompt/history.
- Report when trimming happened.

Required retention/trim order:
- Keep the system message if present, on a best-effort basis.
- Keep the current user turn.
- Drop oldest prior `user` / `assistant` / `tool` turns first.
- Only fail once no more droppable history remains and the request still cannot fit.

Failure behavior:
- If the request still cannot fit after history trimming, raise a clear “input too large even after history trimming” style error.
- Do not misreport this as “increase max_new_tokens” unless the real issue is specifically the requested generation budget.

Implementation allowance:
- Exact token preflight is preferred.
- Acceptable implementations may also use backend-native counters, server-supported truncation controls, or deterministic retry-on-overflow with oldest-turn dropping.
- The exact mechanism is backend-specific; the required outcome is repo-consistent UX.

Reporting:
- Backends must emit/report when history trimming was applied.
- Backends must report if the system message was dropped or could not be preserved.
- If exact counts are knowable, report how many messages were dropped.

## Rollout plan
This spec is an umbrella contract. Implementation should be delivered in separate phased specs/PRs:
1. Token accounting and real tok/s
2. Knob mapping (sent/deferred/ignored)
3. Logging unification (ring buffer parity + `/show logs`)
4. Template control standardization (`chat_template` + capability gating)
