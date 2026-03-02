# 0006 - Support streaming output and configurable think filtering

Date: 2026-02-28

## Context
Reasoning-oriented models may emit long thought blocks before short final answers. This affects both UX and debugging:
- Users want live token feedback (streaming).
- Users may want to hide thought content while preserving useful final output.

## Decision
- Add optional token streaming (`--stream`) to HF runners (`chat.py`, `runner.py`).
- Keep thought filtering as presentation/runtime behavior, not model mutation:
  - `--hide-think`
  - `--strict-think-strip`
- Keep filtering configurable per model via config profiles.

For chat:
- Store filtered assistant text in conversation history when filtering is enabled.

## Consequences
- Better interactive feel for long generations.
- More control over reasoning visibility without requiring model/template surgery.
- Tradeoff: strict filtering can delay visible output if it waits for closing thought markers.

