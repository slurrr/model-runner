#!/usr/bin/env markdown
# Decision: Keep `openai` backend name; add “targets” for multiple servers

Date: 2026-03-08

## Context
We want “vLLM integration”, but the repo’s backend naming convention is by **client protocol / adapter contract**, not by server engine.

We also want to support multiple OpenAI-compatible servers (vLLM local, LM Studio, OpenRouter, etc.) without duplicating endpoint/auth settings across every model config.

## Decision
1) Keep the backend name `openai`.
   - Meaning: “OpenAI-compatible HTTP API adapter”.
   - vLLM is treated as one of several possible servers behind that adapter.

2) Introduce named **targets** for the `openai` backend.
   - Purpose: store server endpoint/auth/timeout in one place.
   - Location: `config/targets/openai/<target>.toml`

3) Maintain model-first configs as “client intent”.
   - Model/prompt/gen/ui remain under `models/<model>/openai/config/default.toml`.

## Consequences
- Users can switch servers without touching per-model config:
  - `--target vllm-local` vs `--target openrouter`
- We avoid “vllm backend” proliferation while still making vLLM the primary recommended server engine.

## Follow-ups
- Spec: `docs/specs/0012-openai-targets-and-server-config-separation.md`

