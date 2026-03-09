# Onboarding Standards (Models + Backends)

This doc defines the standards for adding new backends or extending existing ones so the repo stays consistent as it grows.

Primary goal:
- “Run a model” should feel the same in the TUI regardless of backend, even when the underlying engines differ.

## Model Onboarding (Adding A New Model Folder)

1. Use the scaffold tool
- `python scripts/model add <model_name> <backend> [--id <backend_id>]`

2. Fill in the basics
- `models/<model>/<backend>/notes/README.md`: backend-specific runtime notes (what breaks, what to set, known-good profiles)
- `models/<model>/<backend>/notes/model_card.md`: upstream card (HF README) when applicable
- `models/<model>/<backend>/config/default.toml`: base config (runs out of the box)
- `models/<model>/<backend>/config/profiles/fast.toml`: “go fast” overlay
- `models/<model>/<backend>/config/profiles/longctx.toml`: “go long context” overlay

3. Profiles are overlays, not full copies
- Base config stays readable and stable.
- Profiles should only override what they change.

## Backend Onboarding (Adding A New Engine/Protocol)

### 1. Pick the backend type
Classify before writing code; this determines where standardization is possible.

- In-process engine (local weights, Python API): HF, GGUF, EXL2
- External server attach (OpenAI-compatible): OpenAI backend
- Managed server engine (spawn + attach): vLLM backend
- Vendor-specific server (custom protocol): Ollama backend

### 2. Required implementation surface (TUI)
Every backend must provide:
- `create_session(args)` returning an object that matches `BackendSession` protocol
- `backend_name` and `resolved_model_id`
- `generate_turn(turn_id, messages, emit)` that emits `TurnStart` and ends with either `Finish` or `Error`
- `get_recent_logs(n=80)`

Recommended optional methods:
- `describe()` for `/status` and debugging
- `get_last_request()` for request capture parity when applicable

### 3. Thinking behavior must be consistent in UI

Backend must route output into the same UI affordances:
- Thinking panel receives `ThinkDelta`
- Answer panel receives `AnswerDelta`

Standard approach:
- Use `ThinkRouter(assume_think=args.assume_think)` for any backend that streams plain text.
- If backend provides a dedicated “thinking channel” (e.g., Ollama `message.thinking`, some OpenAI-like servers), emit that via `ThinkDelta` directly and route the rest via `ThinkRouter`.

### 4. Token accounting standard (must)
New backends must implement “token truth” if the backend provides it.

Preferred order of truth:
1. Backend-native usage counters (server usage fields, token IDs from engine)
2. Backend tokenization API (e.g. llama.cpp tokenize)
3. Retokenization fallback (only when a tokenizer is already loaded; avoid loading new heavy deps just for metrics)

At minimum, emit:
- `Meta(prompt_tokens=...)` when knowable
- `Meta(completion_tokens=...)` when knowable
- `Meta(total_tokens=...)` when knowable

### 5. Knob mapping must be explicit
If a backend cannot honor a knob, we must not silently accept it.

Standard requirement:
- Backends report (per turn or once per session):
  - `sent` values (actually applied)
  - `deferred` values (unset, backend default)
  - `ignored` values (user set, backend can’t honor)

This is the only scalable way to avoid “placebo configs”.

### 6. Logging parity
Every backend must:
- support `log_file` in config
- expose `get_recent_logs()` for `/show logs`
- redact secrets in any captured request output

### 7. Config + templates + prompts
Each backend must have a `_TEMPLATE` scaffold at:
- `models/_TEMPLATE/<backend>/config/default.toml`
- `models/_TEMPLATE/<backend>/notes/README.md`
- `models/_TEMPLATE/<backend>/templates/README.md`
- `models/_TEMPLATE/<backend>/prompts/README.md`

If the backend supports local templating:
- provide clear “template source” semantics (tokenizer_config.json vs inline vs override file)
- document any sanitization rules (e.g., don’t re-feed `<think>` into history)

### 8. Documentation requirements
For any non-trivial backend:
- Add a spec under `docs/specs/` for the backend integration contract.
- Add a decision under `docs/decisions/` for any repo-wide change (config format, naming, standard UI behavior).

## Review Checklist (PR Gate)

Before merging a new backend/model addition:
1. `tui` run works from a clean venv with only required deps installed.
2. `TurnStart` and `Finish`/`Error` are always emitted.
3. Thinking routing works with `assume_think` toggles.
4. Token counts are correct or explicitly “unavailable”.
5. Knob mapping is explicit (sent/deferred/ignored).
6. Logs are accessible and secrets are redacted.

