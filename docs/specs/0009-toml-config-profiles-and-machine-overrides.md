# Spec: TOML config (profiles + machine overrides) prep for vLLM/OpenAI backend

Date: 2026-03-07

## Context
Config files in this repo are now doing triple-duty:
- **Model/run settings** (max tokens, temperature, context limits)
- **Backend/engine settings** (dtype, quantization, GGUF engine knobs, Ollama host, EXL2 cache type)
- **Frontend/TUI settings** (follow/scroll, thinking display, transcript saving)

As more backends are added (notably vLLM/OpenAI-compatible servers), mixing these concerns in a single flat JSON object becomes hard to read and easy to misconfigure.

We also want:
- Multiple “virtual builds” per model (fast vs long-context vs agent/tooling)
- A small set of local-only settings (paths, hosts) that should not be committed
- Comments and sectioning for human readability

Python 3.11 includes `tomllib` in the standard library, making TOML a good fit without adding dependencies.

## Goals
- Move to a **human-friendly** config format with:
  - comments
  - sections by concern (model/gen/ui/backend)
- Add **profiles** per model/backend (fast, longctx, agent, etc.).
- Add a **machine-local override** file (not committed) for paths/hosts.
- Preserve existing config precedence and make it explicit.
- Keep backends isolated: each backend only reads its own section/keys.
- Keep a migration path: don’t break existing JSON configs immediately.
- Make it easy to add a new backend (vLLM/OpenAI) without bloating existing configs.

## Non-goals (this iteration)
- Interactive config editing inside the TUI.
- A full schema validation system (we’ll keep it pragmatic: parse + warn on unknown keys).
- A hard cutover that removes JSON support.

## Proposed config layering (merge order)
Precedence from lowest → highest:
1. **Script defaults** (argparse defaults / code defaults)
2. **Backend template defaults** (optional, in-repo)
3. **Model backend default config** (committed, per model/backend)
4. **Profile config** (committed, per model/backend)
5. **Machine override config** (local-only, not committed)
6. **CLI flags** (highest precedence)

Rules:
- For scalar values: last writer wins.
- For lists: last writer wins (no list concatenation).
- For dicts: deep merge (recursive last-writer-wins).
- A value of empty string `""` means “empty string”, not “unset”. (Avoid using it to mean null.)
- Booleans must be overrideable both directions from CLI (`--foo` and `--no-foo`).

## File layout

### Per-model/backend (committed)
Keep model-first layout:
- `models/<model>/<backend>/config/default.toml` (baseline for this backend)
- `models/<model>/<backend>/config/profiles/<name>.toml` (optional profiles)

Example:
- `models/Qwen3.5-9B/hf/config/default.toml`
- `models/Qwen3.5-9B/hf/config/profiles/longctx.toml`
- `models/Qwen3.5-9B/openai/config/default.toml` (future vLLM/OpenAI backend)

### Templates (committed)
- `models/_TEMPLATE/<backend>/config/default.toml`
- `models/_TEMPLATE/<backend>/config/profiles/` (optional example profiles)

### Machine-local overrides (not committed)
Add a top-level local-only config:
- `config/machine.toml` (gitignored)
- `config/machine.example.toml` (committed)

Purpose:
- paths: local model roots, cache dirs
- server endpoints: Ollama host, OpenAI base URL
- local defaults you don’t want copied into the repo history

## TOML schema (sections)
Use section names to make “what applies where” obvious.

### `[model]` (identifiers and paths)
- `id` (string): model identifier for the backend (path or name)
  - HF: local path or HF ID
  - GGUF: `.gguf` file path
  - Ollama/OpenAI: model name on the server
- `display_name` (optional string): purely for UI and notes

### `[gen]` (cross-backend generation knobs)
These are intentionally “generic” and map to different engines where possible:
- `max_new_tokens` (int)
- `max_context_tokens` (int | null)
- `temperature` (float)
- `top_p` (float)
- `top_k` (int | null)
- `min_p` (float | null)
- `typical_p` (float | null)
- `repetition_penalty` (float)
- `presence_penalty` (float | null)
- `frequency_penalty` (float | null)
- `stop_strings` (list[string] | null)
- `seed` (int | null)
- `stream` (bool)

Backend mapping notes:
- Some engines can’t support every knob; the runner must print a one-time “ignored knobs” summary.

### `[prompt]` (system/prefix/template mode)
- `system` (string | optional)
- `system_file` (string | optional, path)
- `user_prefix` (string | optional)
- `prompt_mode` (`"chat"` or `"plain"`, HF/EXL2 primarily)
- `chat_template` (string | optional, path or `default`)

### `[ui]` (TUI-only)
- `show_thinking` (bool)
- `no_animate_thinking` (bool)
- `assume_think` (bool)
- `scroll_lines` (int)
- `ui_tick_ms` (int)
- `ui_max_events_per_tick` (int)
- `save_transcript` (string | empty to disable, path resolved relative to config)

### `[backend.<name>]` (backend-specific)
Examples:
- `[backend.hf]`
  - `dtype` (`auto|float16|bfloat16|float32|torch_auto|checkpoint`)
  - `use_4bit` / `use_8bit` (bool)
  - `attn_implementation` (`eager|sdpa|flash_attention_2`)
- `[backend.gguf]`
  - `n_ctx`, `n_batch`, `n_ubatch`, `n_threads`, `n_gpu_layers`, rope/yarn knobs
- `[backend.ollama]`
  - `host`, `timeout`, `think_mode`
- `[backend.openai]` (future)
  - `base_url`, `api_key` (optional), `timeout`
  - `tool_mode` (future)

### `[machine]` (only in `config/machine.toml`)
Recommended keys:
- `model_root` (e.g. `/home/poop/ml/models`)
- `hf_cache_dir` (optional)
- `ollama_host` (optional)
- `openai_base_url` (optional)

Machine config is *never* required; it only fills blanks.

## CLI integration

### Selection
Keep a single `--config` but add `--profile`:
- `--config` selects a baseline config:
  - path to `.toml` or `.json`
  - or model name (resolved to `models/<model>/<backend>/config/default.*`)
- `--profile <name>` loads `models/<model>/<backend>/config/profiles/<name>.toml` (if present)

Rationale:
- Users keep typing `--config <model>` today; `--profile` is additive and readable.

### Backward compatibility
For a transition period:
- If `default.toml` exists, prefer it.
- Else fall back to `config.json`.
- Support loading JSON explicitly via `.json` extension.

### “Effective config” introspection
Extend `/show config` and `/show gen` to include:
- which files were loaded (default/profile/machine)
- the origin of each shown value (cli/profile/default/machine)
- a backend “payload” view when applicable (what will actually be sent)

## Path resolution rules
- `~` expansion for all paths.
- Relative paths resolve relative to the TOML file that declared them.
- Windows paths (WSL) normalized as currently done in `tui.py`.

## Templates and onboarding
Update `scripts/model add` scaffolding to create:
- `models/<model>/<backend>/config/default.toml`
- optional `profiles/fast.toml` and `profiles/longctx.toml` stubs
- `config/machine.example.toml` once at repo root (if missing)

## Example: HF model config
`models/Qwen3.5-9B/hf/config/default.toml`
```toml
[model]
id = "/home/poop/ml/models/Qwen3.5-9B"

[gen]
stream = true
max_new_tokens = 2048
max_context_tokens = 16384
temperature = 0.7
top_p = 1.0

[prompt]
prompt_mode = "chat"
system = ""
user_prefix = ""

[ui]
show_thinking = true
assume_think = false
save_transcript = ""

[backend.hf]
dtype = "bfloat16"
attn_implementation = "sdpa"
use_4bit = false
use_8bit = false
```

## Example: machine config
`config/machine.example.toml`
```toml
[machine]
model_root = "/home/poop/ml/models"
hf_cache_dir = ""
ollama_host = "http://127.0.0.1:11434"
openai_base_url = "http://127.0.0.1:8000/v1"
```

## Rollout plan
1. Add TOML loading behind existing JSON loader (no breaking changes).
2. Add `config/machine.toml` support (optional, local-only).
3. Add `--profile` and wire profiles resolution.
4. Update templates + `scripts/model add` to generate TOML configs.
5. Migrate a couple models (Nanbeige/Qwen) to TOML to validate the flow.
6. Only then add the OpenAI/vLLM backend spec/implementation that consumes `[backend.openai]`.

## Risks / gotchas
- TOML uses explicit types; avoid ambiguous “empty string means unset” patterns.
- Deep merges must be predictable; document them and keep them simple.
- Some knobs are backend-specific; ensure “ignored knobs” are visible to the user.
- Keep secrets out of committed configs (API keys belong only in `config/machine.toml` or env vars).

