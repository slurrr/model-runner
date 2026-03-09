# AGENTS

## Purpose
This repo is a local LLM playground with a unified terminal UI and multiple backends/engines.

Primary goals:
- Run lots of different local models consistently.
- Keep model-specific notes/config/templates in-repo.
- Make backends comparable and debuggable (thinking routing, logs, token counts, knob mapping).

## Scope
- Keep the project practical and easy to navigate.
- Prefer consistent UX across backends over one-off special cases.
- Favor local-first execution; avoid unnecessary services.
- Notes-first workflow: document behavior before changing code when possible.

## Current Entry Points
- **Unified TUI (recommended):**
  - `tui.py`: multi-backend Textual TUI (scrollback, streaming, thinking panel, slash commands).
  - `tui_app/`: internal package used by `tui.py` (don’t run directly).

- **Simple non-TUI CLIs (useful for isolation/debug):**
  - `chat.py`: HF/Transformers chat loop (template-aware).
  - `runner.py`: minimal HF prompt -> completion loop.
  - `alex.py`: GGUF chat loop (llama.cpp).
  - `ollama_chat.py`: Ollama streaming CLI.

## Conventions
- Prefer straightforward Python and small modules; avoid heavy frameworks.
- Keep backend-specific logic separated under `tui_app/backends/` and shared protocol logic under `tui_app/transports/` when applicable.
- Preserve consistent TUI behavior across backends:
  - streaming events (`TurnStart`, `ThinkDelta`, `AnswerDelta`, `Meta`, `Error`, `Finish`)
  - thinking routing via `ThinkRouter` and `assume_think`
  - `/show` surfaces “sent/deferred/ignored” knobs when available
  - `/show logs` via a per-session ring buffer (no global logs)
- Favor clear failure messages when CUDA/model loading is unavailable.

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

Type prompts at `>` and type `exit` or `quit` to end.

## Repo layout (model-first)
- Model assets live under `models/<model>/<backend>/`:
  - `notes/` (repo-local notes + upstream model card)
  - `config/default.toml` and `config/profiles/*.toml`
  - `templates/` (template overrides where applicable)
  - `prompts/` (usually git-ignored; may contain secrets)

Use `python scripts/model add <model> <backend> [--id <backend_id>]` to scaffold new folders.
