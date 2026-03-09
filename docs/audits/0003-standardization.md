# Backend Standardization Audit (TUI)

Date: 2026-03-08

This audit focuses on the unified Textual TUI path (`tui.py` + `tui_app/`) and compares how each backend behaves today, where we have de-facto standards, and where we should tighten things so adding new backends/models doesn’t re-introduce inconsistency.

Backends in scope:
- HF (`tui_app/backends/hf.py`)
- GGUF (`tui_app/backends/gguf.py`)
- EXL2 (`tui_app/backends/exl2.py`)
- Ollama (`tui_app/backends/ollama.py`)
- OpenAI-compatible (external attach) (`tui_app/backends/openai.py` + `tui_app/transports/openai_http.py`)
- vLLM managed server (`tui_app/backends/vllm.py` + `tui_app/transports/openai_http.py`)

## Current Standard (What’s Already Consistent)

### 1. One event contract for the UI
All TUI backends stream into a shared event model:
- `TurnStart`
- `ThinkDelta`
- `AnswerDelta`
- `Meta` (ad-hoc key/value metadata)
- `Error`
- `Finish` (with a `TurnRecord`)

The minimum “backend session” surface area is standardized via a protocol:
- `backend_name`
- `resolved_model_id`
- `generate_turn(turn_id, messages, emit)`
- `get_recent_logs(n=80)` (exists on the protocol even if some backends return `[]`)

Files:
- `tui_app/backends/base.py`
- `tui_app/events.py`

### 2. One think-splitting primitive
Think vs answer routing is centralized in `ThinkRouter` and used across backends.

What’s standardized:
- Split on marker strings (`<think>`, `</think>`, plus a few alternates)
- `assume_think` support
- Streaming-safe buffer logic for markers split across chunks

File:
- `tui_app/think_router.py`

### 3. Model-first config and folder layout
This repo has converged on:
- `models/<model>/<backend>/config/default.toml`
- `models/<model>/<backend>/config/profiles/<name>.toml` overlays (`--profile <name>`)
- `config/machine.toml` machine-local overrides
- `models/<model>/<backend>/{notes,templates,prompts}/`

Config resolution and TOML flattening is centralized:
- `config_utils.py` (`resolve_config_path`, `load_config_layers`, TOML section flattening)

### 4. “Managed server” reuse
vLLM uses the same OpenAI-compatible HTTP transport as external attach:
- vLLM = “engine that happens to speak OpenAI”
- OpenAI backend = “external server that speaks OpenAI”

Transport:
- `tui_app/transports/openai_http.py`

## Where We’re Not Standardized (And What To Do)

### A. Token accounting and performance metrics (biggest gap)

What we have now:
- HF: can count think tokens using the token-ID streamer (`TokenCountingTextIteratorStreamer`), and emits `Meta(think_tokens_inc=...)`.
- GGUF: can near-exactly count emitted think tokens via `llama_cpp.Llama.tokenize`, emits `Meta(think_tokens_inc=...)`.
- EXL2: has token IDs for chunks and emits `Meta(prompt_tokens=...)`; does not standardize “completion_tokens/total_tokens” in `Meta`.
- Ollama: if `message.thinking` exists, it routes it directly to thinking; otherwise it uses `ThinkRouter` on content; think token count is a whitespace approximation.
- OpenAI/vLLM: think token count is a whitespace approximation; no usage parsing; no prompt/completion token counts.
- `/show last`: prints lengths in *characters* for `raw/think/answer`, not tokens (useful for “is anything happening”, not useful for tok/s).

Impact:
- Tok/s is currently either missing or approximate depending on backend.
- Debugging max context and “thinking ate my budget” becomes guesswork.
- We keep rediscovering “tokens are the unit” but the UI is not instrumented to report them consistently.

Standardization target:
- Every backend should ideally provide:
  - `Meta(prompt_tokens=...)` (int, if knowable)
  - `Meta(completion_tokens=...)` (int, if knowable)
  - `Meta(total_tokens=...)` (int, if knowable)
  - `Meta(tokens_per_s=...)` (float, if completion_tokens is knowable)
- When not knowable, explicitly report:
  - `Meta(token_counts="unavailable")` or a structured `Meta(token_counts={...})`

Concrete standardization opportunities:
1. vLLM/OpenAI transport: parse server-provided usage when enabled and attach to TurnRecord + `/show`.
2. EXL2: emit completion token count and total token count in addition to prompt tokens.
3. Ollama: use Ollama’s own eval counts if available in the response (preferred), fallback to approximation otherwise.
4. `/show last`: display both `chars` and `tokens` when available; don’t pretend chars are tokens.

Decision missing today:
- “Source of truth for token counts”: backend-native usage > backend token IDs > retokenization fallback.

### B. Logging and log tails

What we have now:
- HF/GGUF/EXL2/Ollama/OpenAI use `FileLogger` (file-backed ring buffer).
- vLLM managed uses an in-memory deque for stdout/stderr tail; optional tee to file.

Impact:
- `/show logs` parity is hard until all sessions expose the same “recent logs” behavior.

Standardization target:
- Every backend should expose a recent-log ring buffer (even if it’s filled from file logger).
- Backends that spawn a process should also capture stdout/stderr into that same ring buffer.

### C. “Knob mapping” semantics (sent vs deferred vs ignored)

What we have now:
- HF has native knobs (transformers generate kwargs).
- GGUF has llama.cpp knobs (not all match HF).
- EXL2 has ExLlamaV2 knobs (its own set).
- Ollama only sends options when user explicitly sets them.
- OpenAI transport emits `Meta(ignored_knobs=[...])` for some fields, but doesn’t present a full “sent/deferred/ignored” report.

Impact:
- Users don’t know which knobs actually changed model behavior on that backend.
- Profiles accumulate “placebo knobs” unless we explicitly report what was sent and what was ignored.

Standardization target:
- For every turn, record and optionally show:
  - `sent`: values actually delivered to the engine/server
  - `deferred`: values left to backend defaults (unset)
  - `ignored`: values user set but backend does not support

This should be a TurnRecord-level field (or well-defined `Meta`) so the UI can show it without backend-specific formatting.

### D. Template and history standardization

What we have now:
- HF: uses tokenizer/processor `apply_chat_template`, supports overrides via `--chat-template`.
- EXL2: uses Jinja templates and can read tokenizer_config.json.
- GGUF: uses a chat template spec; fallback to a plain “System/User/Assistant” prompt.
- Ollama/OpenAI/vLLM: template lives server-side; we only send messages.

Impact:
- “Sanitizing history” (e.g., removing `<think>...</think>` before re-sending) is only meaningful when we control templating locally. It’s not universally applicable.

Standardization target (best-effort repo-wide control):
- Aim for repo-controlled chat templates and history shaping whenever feasible:
  - `local_template`: we render prompts/templates (HF, GGUF, EXL2) and can fully sanitize history.
  - `managed_server_template`: we launch the server and can pass template settings at launch (vLLM managed).
  - `server_owned_template`: template is effectively baked into the model/server (Ollama `/api/chat`, arbitrary external OpenAI-compatible servers).
- Only apply history sanitization rules in backends where the repo controls the template (`local_template` and `managed_server_template`).
- Still keep think routing consistent in the UI regardless of template control level.
- Note: attempting “template control” for Ollama-style services is not worth it for this repo; the effective control mechanism is rebuilding the model/server outside this codebase.

### E. Error and timeout semantics

What we have now:
- OpenAI transport uses a total timeout.
- vLLM readiness probe has its own cap.
- Other backends vary (ollama host probing, hf generation errors, gguf context overflow retries, exl2 context restarts).

Standardization target:
- Declare a common error shape in `TurnRecord` (or meta) when available:
  - `stage`
  - `backend`
  - `retryable` (if known)
- Add consistent timeout semantics to config: total vs connect vs read (even if only “total” is supported for v1).

## Where Standardization Is Not Possible (Or Not Worth It Yet)

### 1. Server-driven “thinking” fields
Ollama can stream `message.thinking` separately for supported models; OpenAI-like servers may use `reasoning_content` deltas. Local backends don’t have this.

We can standardize the *UI experience* (thinking panel + toggle), but we should not force a single transport contract across all sources of thinking.

### 2. Full knob parity
Sampling and engine knobs differ across transformers/llama.cpp/exllamav2/vLLM/Ollama.

We can standardize:
- naming in config (`[gen]` is stable)
- reporting (sent/deferred/ignored)

But we cannot standardize “every knob exists everywhere” without turning the repo into an abstraction tax.

## Recommended Standardization Roadmap (Minimal, High-Leverage)

1. Token counts and tok/s
- vLLM: parse usage tokens when enabled.
- EXL2: emit completion and total tokens.
- GGUF: add prompt/completion/total (it can tokenize).
- HF: add prompt token count (we already compute prompt length internally).

2. Best-effort chat template and history control
- Standardize a `chat_template`-style knob across backends that can honor it:
  - HF/GGUF/EXL2: template is repo-owned and applied locally.
  - vLLM managed: template is repo-owned and passed at server launch.
  - Ollama: explicitly excluded (server-owned template via Modelfile/build).
- Add/standardize template docs and a consistent “history sanitization” policy for repo-owned templates.

3. Per-turn “sent/deferred/ignored” reporting
- Make this a standard TurnRecord field (or a structured Meta key).
- Make `/show gen` default output depend on this instead of ad-hoc printing.

4. Standard log tail
- Ensure every session’s `get_recent_logs()` returns something meaningful.
- Pipe managed server logs into that same surface.
- Adopt the TUI log buffer contract in `docs/audits/0004_tui_logging.md` (session-owned ring buffer, optional file sink, UTC timestamps, redaction).

5. Capabilities declaration
- Add a backend capability dict exposed via `describe()` or `capabilities()`:
  - `supports_images`
  - `template_control_level`
  - `supports_usage_tokens`
  - `supports_tool_calls` (transport-level)

## Notes For Model Onboarding

The biggest “repo drift” risk isn’t adding more models; it’s adding more backends without:
- explicit capability reporting
- explicit knob mapping (“this setting actually did something”)
- explicit token accounting

If those three are standardized, new backends/models become predictable to reason about and compare.
