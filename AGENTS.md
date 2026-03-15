# AGENTS

## Purpose
This repo is a local model lab for:
- local model research
- runtime tuning and backend comparison
- agent and application experimentation
- model-specific notes/config/template management

It spans both:
- inference/runtime engineering
- AI application and agent development testing

Treat it as a practical research workspace, not just a terminal chat app.

Primary goals:
- run lots of local models consistently across multiple backends
- make runtime behavior visible enough to learn from and compare
- keep model-specific knowledge in-repo next to the configs/templates that matter
- make backend experimentation fast, local, and repeatable

## Priority Order
When making engineering tradeoffs, prefer:
1. runtime truth and observability
2. comparable behavior across backends
3. local-first experimentation speed
4. enjoyable interaction surfaces

If a UX summary conflicts with backend truth, fix the summary instead of hiding the truth.

## Scope
- Keep the project practical and easy to navigate.
- Favor local-first execution; avoid unnecessary services.
- Preserve backend diversity instead of forcing one engine to define the whole repo.
- Prefer standardized reporting where possible, but do not flatten away backend-specific facts that matter for tuning.
- Notes-first workflow: document behavior before changing code when possible.

## Current UX Posture
- The Textual TUI is a valid current control surface.
- The TUI is not the final observability or optimization product.
- Work that improves browser/dashboard-oriented observability is aligned with repo direction.
- Terminal UX should remain useful, but new designs should not assume the TUI is the only long-term interface.

## Current Entry Points
- **Unified TUI (current interaction surface):**
  - `tui.py`: multi-backend Textual TUI
  - `tui_app/`: internal package used by `tui.py` (don’t run directly)

- **Managed backend control:**
  - `vllm-up`, `vllm-down`, related repo helpers for managed vLLM bring-up

- **Simple non-TUI CLIs (good for isolation/debug):**
  - `chat.py`: HF/Transformers chat loop
  - `runner.py`: minimal HF prompt -> completion loop
  - `alex.py`: GGUF chat loop
  - `ollama_chat.py`: Ollama streaming CLI

## Conventions
- Prefer straightforward Python and small modules; avoid heavy frameworks unless they materially improve the lab.
- Keep backend-specific logic under `tui_app/backends/` and shared protocol logic under `tui_app/transports/` when applicable.
- Preserve consistent event behavior across backends:
  - `TurnStart`, `ThinkDelta`, `AnswerDelta`, `Meta`, `Error`, `Finish`
  - thinking routing via `ThinkRouter` and `assume_think`
  - `/show` surfaces for effective/requested state when available
  - `/show logs` via a per-session ring buffer
- Prefer runtime facts over intent-only displays.
- Favor clear failure messages when CUDA/model loading/runtime support is unavailable.

## Usage
Recommended:
```bash
python tui.py <model> [backend_hint]
python tui.py --config <model_name> --backend <hf|gguf|ollama|exl2|openai|vllm> [--profile <name>]
```

Legacy/simple:
```bash
python runner.py <hf_model_or_local_path>
python chat.py <hf_model_or_local_path>
python alex.py <path_to_model.gguf>
python ollama_chat.py <ollama_model_name>
```

## Repo Layout (model-first)
- Model assets live under `models/<model>/<backend>/`:
  - `notes/` for repo-local notes and upstream model cards
  - `config/default.toml` and `config/profiles/*.toml`
  - `templates/` for template overrides
  - `prompts/` for prompt/system assets

Use:
```bash
python scripts/model add <model> <backend> [--id <backend_id>]
```
to scaffold new folders.

## Direction
The repo is moving toward a stronger local observability story:
- better backend/runtime truth surfaces
- better comparison across engines
- browser/dashboard-oriented metrics and visualization
- a more complete local model lab feel

The observability direction already has a home in:
- `docs/observability_dashboard`

Keep that direction in mind when making UX or architecture decisions.
