# 0007 - Adopt model-first `models/` workspace

Date: 2026-03-01

## Context
As the repo grew (notes, configs, templates, prompts, specs/audits), backend-first trees became harder to navigate during day-to-day use. The primary workflow is “I’m working with *this model* right now”, often across multiple backends (HF + GGUF + Ollama).

## Decision
Adopt a model-first workspace under `models/`, with backend separation inside each model:

- `models/<model>/<backend>/notes/`
- `models/<model>/<backend>/config/`
- `models/<model>/<backend>/templates/`
- `models/<model>/<backend>/prompts/`

Shared assets live under:
- `models/_shared/<backend>/templates/`
- `models/_shared/<backend>/prompts/`

Template starters live under:
- `models/_TEMPLATE/<backend>/...`

## Consequences
- Navigation is model-centric and scales as you add more models and backends.
- `--config <model>` remains intuitive: HF runners resolve short names to `models/<model>/hf/config/config.json` (and still accept direct file paths).
- Older backend-first folders (`config/`, `model_notes/`, `templates/`) are removed to avoid duplicate sources of truth.

