# 0005 - Add per-model config profiles with CLI override precedence

Date: 2026-02-28

## Context
Per-model runtime settings are now substantial (dtype, token limits, sampling, think filtering, streaming, prefixes, etc.). Repeating long command lines is error-prone and slows testing.

## Decision
Introduce model-first config profiles under `models/<model>/<backend>/config/` and add `--config` support in HF runners.

Current support:
- `chat.py --config ...`
- `runner.py --config ...`

Resolution behavior:
- Accept direct file path or path without `.json`
- Accept short names resolved under `models/<name>/hf/config/config.json`

Precedence:
1. CLI flags
2. Config values
3. Script defaults

Also add a reusable template profile:
- `models/_TEMPLATE/hf/config/config.json`

## Consequences
- Model runs are reproducible and easier to share.
- New model onboarding becomes “copy template, adjust a few keys”.
- Backward compatibility is preserved because positional args and direct flags still work.
