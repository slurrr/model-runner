#!/usr/bin/env markdown
# Decision: Add `/status` and make `/show` the primary inspector (with hidden aliases)

Date: 2026-03-08

## Context
Slash commands grew organically. We need:
- a concise “what’s going on?” view for normal use
- a consistent deep-inspection surface for debugging
- fewer primary commands in help, without breaking old muscle memory

## Decision
- Add `/status` as the default concise runtime summary.
- Keep `/show` as the primary deep inspection command (topics + `--verbose`).
- Keep existing shortcut commands as hidden aliases mapping to `/show <topic>` for backward compatibility.
- Add `/show logs` (and optionally `/show request`) to enable managed-mode debugging without leaving the TUI.
- Standardize backend log introspection with a shared in-memory tail contract:
  - backend sessions expose `get_recent_logs(n=80) -> list[str]`
  - `/show logs` reads this method across backends
  - file logging remains optional and separate from in-memory tails

## Consequences
- Help output becomes cleaner and more navigable.
- Debugging managed vLLM becomes feasible without rerunning in “external attach” mode.
- Debugging is more consistent across backends, even when file logging was not enabled.
- We can later add session save/restore without multiplying commands.

## Spec
- `docs/specs/0014-tui-status-and-show-inspect.md`
