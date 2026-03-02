# 0004 - Organize documentation under `docs/`

Date: 2026-03-01

## Context
The repo now includes decision records, audits, and implementation specs in addition to scripts. Keeping these at the repo root makes navigation harder and increases noise as the project grows.

## Decision
Create a `docs/` hierarchy:
- `docs/decisions/` for ADR-style decisions
- `docs/specs/` for implementation specs
- `docs/audits/` for investigations/audits of current behavior

Keep model-specific, operational “what to run / what to expect” notes/config/prompts/templates in `models/` (not under `docs/`).

## Consequences
- Root stays focused on runnable entrypoints and core config.
- Docs are easier to discover and evolve without cluttering the top level.
