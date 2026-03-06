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
