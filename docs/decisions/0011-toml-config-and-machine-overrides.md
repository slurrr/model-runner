# 0011 - Adopt TOML configs with profiles and machine-local overrides

Date: 2026-03-07

## Context
Config in this repo has grown to include:
- generation knobs
- backend/engine knobs
- TUI/UI knobs

Flat JSON is machine-friendly but has become hard to read and hard to distinguish “model vs backend vs UI” settings. We also want local-only settings (paths, hosts, API keys) that should not be committed, and multiple runnable “profiles” per model (fast/agent/longctx).

Python 3.11 includes `tomllib`, so TOML provides comments + sections without adding dependencies.

## Decision
- Primary config format becomes **TOML** with sections by concern:
  - `[model]`, `[gen]`, `[prompt]`, `[ui]`, and `[backend.<name>]`
- Support **profiles** per model/backend under:
  - `models/<model>/<backend>/config/profiles/<name>.toml`
- Support an optional **machine-local override** file:
  - `config/machine.toml` (gitignored), with a committed `config/machine.example.toml`

Backward compatibility:
- Continue supporting existing JSON configs for a transition period.
- Prefer `default.toml` if present; otherwise fall back to `config.json`.

This extends (and partially supersedes) decision `0005` by preserving the idea of per-model config profiles while switching the primary format to TOML.

## Consequences
- Configs are easier to navigate and comment.
- Model settings and backend settings become visually separated.
- Adding new backends (e.g. vLLM/OpenAI-compatible servers) doesn’t bloat existing configs.
- Some implementation work is required: a loader/merger and updated scaffolding templates.

