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

## Format

- One file per decision: `0001-<slug>.md`, `0002-<slug>.md`, ...
- Keep them practical: context → decision → consequences.
- Use dates in ISO format (YYYY-MM-DD).
