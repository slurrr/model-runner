# Specs

Implementation specs for features we plan to add (kept lightweight and pragmatic).

- `docs/specs/0001-textual-tui.md`: Initial HF-only Textual TUI spec.
- `docs/specs/0002-unified-tui-entrypoint.md`: Unified `tui.py` entrypoint across HF + GGUF + Ollama.
- `docs/specs/0003-assume-think-prefix.md`: Minimal `assume_think` routing for models that emit only `</think>`.
- `docs/specs/0004-tui-slash-commands.md`: Registry-driven slash commands for runtime inspection (`/show`, `/system`, `/?`, ...).
- `docs/specs/0005-gguf-chat-templates.md`: GGUF client-side chat templates (no rebuild) via llama.cpp handlers.
- `docs/specs/0006-gguf-engine-knobs.md`: Expose GGUF engine knobs (batch/threads/RoPE/YARN).
- `docs/specs/0007-exl2-backend.md`: Add ExLlamaV2 (EXL2) as a TUI backend (client-side templates).
- `docs/specs/0008-reviews-branch-mirror.md`: Mirror PR review bot output into a dedicated `reviews` branch for local/offline consumption.
- `docs/specs/0009-toml-config-profiles-and-machine-overrides.md`: TOML config format with profiles + machine-local overrides (prep for vLLM/OpenAI backend).
- `docs/specs/0010-openai-compatible-backend-vllm.md`: Add an OpenAI-compatible HTTP backend (vLLM first) and document the “backend build” workflow.
- `docs/specs/0011-openclaw-tool-compat-layer.md`: OpenAI tool schema loading + policy + tool-call normalization (OpenClaw-oriented).
- `docs/specs/0012-openai-targets-and-server-config-separation.md`: Deferred reference spec for OpenAI “targets” and clearer client/server config separation.
- `docs/specs/0013-vllm-engine-first-managed-server.md`: Make vLLM a first-class engine backend (managed start/stop).
- `docs/specs/0014-tui-status-and-show-inspect.md`: `/status` + a cleaner `/show` inspector UX for the TUI.
- `docs/specs/0015-backend-standardization-contracts.md`: Umbrella contract for backend consistency (thinking, metrics, logging, knobs, templates).
- `docs/specs/0016-token-accounting-and-real-toks.md`: Phase 1 spec to add real token counts and tok/s across backends.
- `docs/specs/0017-tui-logging-ring-buffer.md`: Standardize session-owned in-memory log tail across all backends.
- `docs/specs/0018-generation-knob-reporting.md`: Standardize per-turn reporting of sent/deferred/ignored generation knobs.
- `docs/specs/0019-chat-template-and-history-control.md`: Best-effort repo-wide chat template control + think-stripping for history.
- `docs/specs/0020-tui-mvp-tool-execution-harness.md`: Minimal, safe tool execution loop in the TUI (vLLM/OpenAI first).
- `docs/specs/0021-safe-browser-tool-v1.md`: Fetch-only browser tool with SSRF + injection guardrails.
- `docs/specs/0023-local-observability-dashboard.md`: Repo-owned local observability product with a web dashboard, canonical telemetry model, and future same-surface web chat path.
