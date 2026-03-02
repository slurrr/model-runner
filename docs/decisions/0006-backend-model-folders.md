# 0006 - Standardize backend/model folder layout

Date: 2026-03-01

## Context
We organize several repo areas by backend (`hf`, `gguf`, `ollama`). As the repo grows, a flat list of files per backend becomes hard to navigate and doesn’t leave room for per-model supporting artifacts (tools JSON, system prompts, alternate templates, audit snippets, etc.).

## Decision
Standardize on a model-first layout with backend separation:
- `models/<model>/<backend>/notes/README.md`
- `models/<model>/<backend>/config/config.json`
- `models/<model>/<backend>/templates/...`
- `models/<model>/<backend>/prompts/...`

Shared assets live under:
- `models/_shared/<backend>/templates/`
- `models/_shared/<backend>/prompts/`

## Consequences
- Adding per-model auxiliary files becomes natural (e.g. `tools.json`, `system.txt`, template overrides).
- `--config` resolution should support both file paths and model folders (handled by `config_utils.py`).
- Documentation and examples should refer to model folders instead of single JSON files.
