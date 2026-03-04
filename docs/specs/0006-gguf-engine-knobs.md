# Spec: expose GGUF “engine knobs” (batch/threads/RoPE/YARN)

Date: 2026-03-03

## Problem
For GGUF (llama.cpp via `llama-cpp-python`), a lot of “fun” and practically important performance/memory controls live outside of the usual
sampling knobs (temperature/top_p/etc.). Today the unified TUI exposes only a small subset for GGUF:
- load-time: `n_ctx`, `n_gpu_layers`
- generation-time: `temperature`, `top_p` (and stop strings)

For a local LLM playground, we want to expose additional llama.cpp knobs so we can:
- fit large models tightly into VRAM
- experiment with throughput/latency (prompt eval vs token streaming)
- explore context scaling knobs (RoPE/YARN) without rebuilding

## Goal
Expose a “useful set” of GGUF engine knobs via:
- unified TUI CLI flags (`tui.py`)
- per-model GGUF config (`models/<model>/gguf/config/config.json`)
- pass-through into `tui_app/backends/gguf.py` (load-time) and generation calls (runtime)

This spec intentionally focuses on knobs that are:
- commonly referenced in llama.cpp discussions
- relatively safe to tweak
- helpful for “fit into VRAM” experimentation

## Scope (v1 knobs to add)

### Load-time (Llama constructor)
Add to config + CLI + `Llama(...)`:

- `n_batch` (int, default: llama.cpp default or config default)
- `n_ubatch` (int, default: llama.cpp default or config default)
- `n_threads` (int|null)
- `n_threads_batch` (int|null)

RoPE/YARN (context scaling controls):
- `rope_scaling_type` (int / enum-ish, default: -1)
- `rope_freq_base` (float)
- `rope_freq_scale` (float)
- `yarn_ext_factor` (float)
- `yarn_attn_factor` (float)
- `yarn_beta_fast` (float)
- `yarn_beta_slow` (float)
- `yarn_orig_ctx` (int)

Other engine toggles:
- `flash_attn` (bool|null) (if supported by your build)
- `offload_kqv` (bool|null)

Repro/debug:
- `seed` (int|null) for GGUF sessions (see “Seed semantics”)

### Generation-time (create_chat_completion / create_completion)
GGUF already uses `create_chat_completion(stream=True)` and a `create_completion(...)` fallback.
We should also pass through (when set / non-default):

- `top_k` (int|null)
- `min_p` (float|null)
- `typical_p` (float|null)
- `repeat_penalty` (float; map from config `repetition_penalty`)
- `presence_penalty` (float|null)
- `frequency_penalty` (float|null)
- `tfs_z` (float|null)
- `mirostat_mode` (int|null)
- `mirostat_tau` (float|null)
- `mirostat_eta` (float|null)
- `seed` (int|null) (per-request seed)

Note: some of these already exist as unified flags (e.g. `min_p`, `typical_p`, `top_k`, `repetition_penalty`) but are not forwarded by
`tui_app/backends/gguf.py` today.

## CLI design

### Naming
Prefer llama.cpp naming for load-time knobs to reduce confusion:
- `--n-ctx`, `--n-batch`, `--n-ubatch`, `--n-threads`, `--n-threads-batch`, `--n-gpu-layers`

RoPE/YARN:
- `--rope-scaling-type`
- `--rope-freq-base`
- `--rope-freq-scale`
- `--yarn-ext-factor`
- `--yarn-attn-factor`
- `--yarn-beta-fast`
- `--yarn-beta-slow`
- `--yarn-orig-ctx`

Toggles (tri-state to allow “config true/false” and “unset means backend default”):
- `--flash-attn` / `--no-flash-attn`
- `--offload-kqv` / `--no-offload-kqv`

Seed:
- `--seed <int>`

### Boolean precedence
Follow the existing “tri-state CLI overrides config” approach:
- CLI defaults to `None`
- if user passes explicit flag, override config
- otherwise use config value

## Config design

### Add keys to GGUF config template
Update `models/_TEMPLATE/gguf/config/config.json` to include the new keys with conservative defaults:

- Use `null` for optional/tri-state toggles (means “don’t override backend”).
- Use explicit numbers only where a safe default exists and is commonly used in this repo.

### Where values apply
- Load-time knobs apply when the model loads (TUI startup).
- Sampling knobs apply per-turn.

## Seed semantics
Two places can take a seed:
- model init seed (`Llama(..., seed=...)`) for some internal randomness
- per-request seed (`create_chat_completion(..., seed=...)`)

Recommended:
- if `seed` is provided, pass it to both init and request for maximum reproducibility
- document that full determinism is not guaranteed across different hardware/builds

## Implementation steps

1) `tui.py`
   - Add CLI flags and include them in `_collect_config_defaults(...)`.
   - Ensure `_warn_ignored_flags(...)` does not warn for GGUF-relevant knobs.

2) `tui_app/backends/gguf.py`
   - Pass new load-time knobs into `Llama(...)`.
   - Pass new generation-time knobs into `create_chat_completion(...)` and the fallback `create_completion(...)`.
   - Print a short “GGUF engine” summary at startup (optional, but helpful for verification).

3) Templates / docs
   - Update `models/_TEMPLATE/gguf/config/config.json`.
   - Update `docs/tuning.md` with a GGUF section describing what these knobs do.

## Testing checklist

- Startup works with no extra flags (backward compatible).
- `--n-batch` and `--n-ubatch` affect performance without breaking streaming.
- `--n-threads` and `--n-threads-batch` apply without error.
- RoPE/YARN values pass through and do not crash (quality is model-dependent).
- `--seed` produces repeatable-ish outputs for a fixed prompt/sampler.

