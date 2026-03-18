#!/usr/bin/env markdown
# Spec: OpenAI-compatible targets + client/server config separation

Date: 2026-03-08

Status: **Deferred / future** (kept for reference)

## Problem
The repo has an `openai` backend (OpenAI-compatible HTTP protocol adapter). We want to use it primarily with vLLM, but also with other OpenAI-compatible servers later.

Two pain points:
1) Naming confusion: “vLLM integration” vs “openai backend”.
2) Config shape confusion: mixing *client intent* (model/prompt/gen) with *server endpoint/auth* (base_url/key/timeout), especially when you have multiple servers.

## Why deferred
We decided to make **vLLM a first-class engine backend** (managed start/stop) instead of making “OpenAI-compatible HTTP” the primary user-facing backend.

That direction is specified in:
- `docs/specs/0013-vllm-engine-first-managed-server.md`

This spec remains useful if/when we add:
- “external attach” mode (connect to an already-running vLLM)
- support for remote OpenAI-compatible servers (OpenRouter/LM Studio/etc)
- named transport “targets” that can be shared across models/engines

## Goals
- Keep the backend named by **protocol contract** (`openai`), while making it easy to target vLLM and other OpenAI-compatible servers.
- Reduce duplication when you have many models but only a few server endpoints.
- Make it obvious which settings are:
  - model-level / prompt-level / generation-level (client intent)
  - transport/auth (client-to-server)
  - server launch knobs (server-side; out of scope for the TUI client)

## Non-goals (v1)
- Starting/stopping vLLM from the TUI.
- Capturing every vLLM CLI flag in repo config.

## Terminology
- **Backend (repo)**: adapter that emits the TUI streaming events. Example: `hf`, `gguf`, `ollama`, `exl2`, `openai`.
- **Server engine**: the thing actually running the model. Example: vLLM (HF models), LM Studio, OpenRouter, etc.
- **Target**: a named OpenAI-compatible server endpoint + auth settings.

## Current state (today)
- Model config lives under: `models/<model>/<backend>/config/default.toml`
- Optional per-model profiles live under: `models/<model>/<backend>/config/profiles/<name>.toml`
- Machine defaults (gitignored) live at: `config/machine.toml`

This already supports targeting multiple servers, but only by duplicating profiles per model (or constantly passing CLI flags).

## Proposed change: add global “targets”
Add a repo-level folder for server targets:
- `config/targets/openai/<target>.toml`

Each target file contains only transport/auth keys:
```toml
[backend.openai]
base_url = "http://127.0.0.1:8000/v1"
api_key = ""
timeout_s = 600
```

Rules:
- Targets contain **no** model/prompt/gen/ui keys.
- Secrets still belong in `config/machine.toml` (gitignored) unless you explicitly want them in a target file.

## Config layering order (MUST)
For `--backend openai`, merge layers in this order:
1) Base: `models/<model>/openai/config/default.toml`
2) Optional per-model profile: `models/<model>/openai/config/profiles/<profile>.toml`
3) Optional target: `config/targets/openai/<target>.toml`
4) Optional machine: `config/machine.toml`
5) CLI flags (highest precedence)

Rationale:
- Per-model profile should override the base model defaults.
- Target should allow switching servers without rewriting per-model configs.
- Machine should set per-machine defaults without committing secrets.
- CLI always wins.

Note: If we want “machine overrides win” semantics later, we can swap (3) and (4). v1 chooses “target beats machine” so you can quickly switch endpoints without editing machine.toml.

## CLI changes (MUST)
Add:
- `--target` (string): resolves to `config/targets/openai/<target>.toml` for the `openai` backend.

Examples:
- `tui Qwen3.5-9B --backend openai --target vllm-local`
- `tui Qwen3.5-9B --backend openai --target openrouter`

If `--target` is set but the file doesn’t exist, fail with a clear error listing:
- expected path
- existing targets in `config/targets/openai/` (if any)

## Recommended repo layout conventions
- Keep model “intent” in model config:
  - `[model].id` (server-side model name)
  - `[prompt]` system/system_file/user_prefix
  - `[gen]` temperature/top_p/max_new_tokens/stop_strings/seed…
  - `[ui]` show_thinking, assume_think, save_transcript…
- Keep endpoint/auth in targets + machine:
  - `[backend.openai]` base_url/api_key/timeout_s
- Keep server launch instructions as docs:
  - `docs/environments/vllm.md` (optional, not required for v1)

## vLLM “isn’t a backend” clarification (docs requirement)
Docs/entrypoints MUST clearly state:
- The `openai` backend is a protocol adapter.
- vLLM is the recommended **server engine** that exposes that protocol locally.

Optionally: add an alias in CLI (`--backend vllm` == `--backend openai`) later, but do not rename the backend.

## Open questions
- Do we want targets for other backends too (`config/targets/ollama/...`), or keep targets OpenAI-only for now?
- Should machine config be allowed to set a default target name (e.g. `openai_target_default = "vllm-local"`)?
