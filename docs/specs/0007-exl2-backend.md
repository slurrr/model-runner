# Spec: ExLlamaV2 / EXL2 backend for unified TUI

Date: 2026-03-03

## Problem
EXL2 (ExLlamaV2) models are fast and useful for local experimentation, but this repo currently supports only:
- HF/Transformers (folder checkpoints)
- GGUF (llama.cpp)
- Ollama API

Ollama template iteration is slow (rebuild required). We want a **TUI-only** EXL2 backend with:
- client-side templates stored in `models/<model>/exl2/templates/`
- per-model config stored in `models/<model>/exl2/config/config.json`
- good error messages for common CUDA/extension build mismatches

## Goals
- Add backend: `exl2` to `tui.py` (select via `--backend exl2` or `exl2:<path>` prefix).
- Load EXL2 model directories and stream tokens into the existing Textual UI.
- Use repo-local Jinja templates to build the prompt (no rebuild loop).
- Support basic “engine” knobs commonly used in ExLlamaV2:
  - `gpu_split` (including `auto` autosplit)
  - `max_seq_len` / sequence length
  - attention toggles: flash-attn / xformers / sdpa / graphs (tri-state where sensible)
  - cache type (fp16 vs 8bit/q4/q6/q8)
- Support key sampling knobs via `ExLlamaV2Sampler.Settings`:
  - temperature/top_k/top_p/min_p/typical
  - repetition/frequency/presence penalties

Non-goals (v1):
- tool-call schema enforcement / JSON mode
- multi-modal EXL2
- LoRA wiring and speculative decoding

## UX

### Selecting backend
Because EXL2 models are directories (like HF), backend auto-detection is ambiguous.

Supported selection:
- `tui --backend exl2 /path/to/model_dir`
- `tui exl2:/path/to/model_dir` (optional convenience)

Default remains HF for bare directory paths.

### Templates
Templates live in:
- `models/<model>/exl2/templates/`

Config/CLI:
- `chat_template` supports:
  - empty: try `tokenizer_config.json`’s `chat_template` if present; else fallback to a minimal built-in template
  - file path: use that file (Jinja2)
  - JSON file path: read `chat_template` / `template` key (HF-style)

Template context variables (HF-style subset):
- `messages` (list of `{role, content}`)
- `bos_token`, `eos_token` (strings from ExLlamaV2Tokenizer)
- `add_generation_prompt` (bool; default true)

### Stop strings
- ExLlamaV2 streaming generator supports string and token stop conditions.
- Map `stop_strings` from config/CLI directly to `generator.set_stop_conditions([...])`.

### “Thinking” tags
- EXL2 backend output is routed through `ThinkRouter` (existing).
- Model output is not modified; `assume_think` is supported as a UI routing knob.

## Config

Model config path:
- `models/<model>/exl2/config/config.json`

Key fields (v1):
- `model_id` (required): local EXL2 model directory
- `max_new_tokens`
- sampling: `temperature`, `top_k`, `top_p`, `min_p`, `typical`, `repetition_penalty`, `frequency_penalty`, `presence_penalty`
- prompt: `system`, `system_file`, `user_prefix`, `chat_template`
- context/engine: `max_seq_len`, `gpu_split`, `cache_type`
- attention/engine toggles: `flash_attn`, `xformers`, `sdpa`, `graphs` (tri-state)
- misc: `seed` (best-effort; ExLlamaV2 is not guaranteed deterministic)

## Implementation

### New backend adapter
Add: `tui_app/backends/exl2.py`

Responsibilities:
- resolve model path (WSL Windows path normalization)
- load ExLlamaV2 classes lazily (optional dependency)
- build prompt using template + messages
- encode with `tokenizer.encode(..., encode_special_tokens=True)`
- trim history if encoded prompt exceeds `max_seq_len - min_space`
- generate via `ExLlamaV2StreamingGenerator`:
  - `begin_stream_ex(input_ids, settings)`
  - loop `stream_ex()` until `eos` or max tokens
  - emit deltas into the TUI event stream

### Glue in `tui.py`
- Add backend choice: `exl2`
- Add config defaults support for new keys
- Add `create_exl2_session(...)`
- Detect `exl2:` prefix

### Model scaffolding
Add backend template folder:
- `models/_TEMPLATE/exl2/config/config.json`
- `models/_TEMPLATE/exl2/notes/README.md`

Update `scripts/model`:
- allow backend `exl2`
- `--id` should prefill `model_id` for `exl2` (similar to `hf`)

## Installation / CUDA mismatch notes (must-have docs)

EXL2 requires a compiled extension (`exllamav2_ext`) built by PyTorch’s extension tooling.
Common failure modes:
- missing `ninja`
- `nvcc` toolkit version mismatch with `torch.version.cuda`
- stale extension cache under `~/.cache/torch_extensions/`

Doc requirements:
- Quick checks:
  - `python -c "import torch; print(torch.__version__, torch.version.cuda)"`
  - `nvcc --version`
- Guidance:
  - Ideally, `torch.version.cuda` major.minor matches `nvcc` major.minor.
  - If mismatched, either:
    - install a matching CUDA toolkit (preferred), or
    - install a torch build matching your installed CUDA toolkit.
- Provide “clean rebuild” steps for the extension cache.

