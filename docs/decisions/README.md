# Decisions

This folder stores short decision records (ADRs) for choices made while evolving this repo beyond “just a small runner”.

## Index

| ID | Title | Summary |
|---|---|---|
| 0001 | Capture per-model notes in-repo | Store model behavior notes under `models/<model>/<backend>/notes/`. |
| 0002 | Treat decoding knobs as `transformers`-level, not model-level | Separate checkpoint defaults from generic generation controls. |
| 0003 | Notes-first workflow and explicit approval for code changes | Default to documenting model findings before script edits. |
| 0004 | Keep backends separated by script | Use dedicated entrypoints for HF, GGUF, and Ollama backends. |
| 0005 | Add per-model config profiles with CLI override precedence | Add `models/<model>/<backend>/config/` profiles and `--config` support with clear precedence. |
| 0006 | Support streaming output and configurable think filtering | Add `--stream`, `--hide-think`, and `--strict-think-strip` controls. |
| 0007 | Adopt model-first `models/` workspace | Organize notes/config/templates/prompts under `models/<model>/<backend>/...`. |
| 0008 | TUI follow mode is intent-driven | Follow state changes only from explicit user scroll intent, not passive scroll-delta inference. |
| 0009 | TUI slash commands are local-only and registry-driven | Provide in-TUI introspection without sending commands to the model. |
| 0010 | Mirror PR review output into a `reviews` branch | CI writes bot review bundles to an orphan `reviews` branch for local/offline reading. |
| 0011 | Adopt TOML configs with profiles and machine-local overrides | Switch primary config to TOML sections; add profiles and local-only machine overrides. |
| 0012 | Store upstream model cards as `notes/model_card.md` | Keep upstream HF README content separate from repo-local notes. |
| 0013 | Keep `openai` backend name; add “targets” for multiple servers | Name backends by protocol contract; add `config/targets/openai/` for endpoints/auth. |
| 0014 | Add `/status`; make `/show` the primary inspector | Provide concise status + verbose inspection; keep old shortcuts as hidden aliases. |
| 0015 | Token accounting and metrics | Prefer backend-native usage/token IDs; fall back to backend tokenization; avoid char-as-token substitutions. |

## Format

- One file per decision: `0001-<slug>.md`, `0002-<slug>.md`, ...
- Keep them practical: context → decision → consequences.
- Use dates in ISO format (YYYY-MM-DD).
