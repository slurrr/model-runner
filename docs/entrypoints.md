# Entrypoints

This repo intentionally has a few small “front door” scripts. Pick the one that matches what you’re trying to do.

## Recommended (default): unified TUI

Use `tui.py` when you want a terminal UI (scrollback, collapsible thinking, streaming) and/or you want to switch between backends with one command.

Optional: install a console entrypoint so you can run `tui ...` instead of `python tui.py ...`:
```bash
pip install -e .
```

What it supports:
- HF / Transformers (local folder or HF id)
- GGUF via llama.cpp (`.gguf` path; requires `llama-cpp-python`)
- Ollama (`ollama:<name>`; requires Ollama running)

Examples:
```bash
# HF (local folder under ~/ml/models/ or an HF id)
tui Nanbeige4.1-3B
python tui.py Nanbeige4.1-3B
python tui.py --config Nanbeige4.1-3B

# GGUF (auto-detected by .gguf extension)
tui /mnt/d/models/your-model.gguf
python tui.py /mnt/d/models/your-model.gguf

# Ollama (explicit by scheme prefix)
tui ollama:your-ollama-model
python tui.py ollama:your-ollama-model
python tui.py ollama:your-ollama-model --ollama-think false
```

Notes:
- `tui_app/` is an internal package used by `tui.py` (don’t run `tui_app/app.py` directly).
- Backend detection is deterministic:
  - `ollama:<name>` → Ollama
  - `*.gguf` → GGUF
  - existing directory → HF
  - otherwise → HF
- When running under WSL, Windows paths like `D:\models\foo.gguf` are normalized to `/mnt/d/models/foo.gguf`.

## HF CLI chat (template-aware)

Use `chat.py` when you want a simple terminal chat loop (no TUI) and you’re working with HF/Transformers chat models and templates.

Examples:
```bash
python chat.py Nanbeige4.1-3B
python chat.py --config Nanbeige4.1-3B

# Force plain prompting (no chat template)
python chat.py Nanbeige4.1-3B --prompt-mode plain

# Precision / quantization (CUDA required for 4/8-bit)
python chat.py Nanbeige4.1-3B --dtype bfloat16
python chat.py Nanbeige4.1-3B -8bit
python chat.py Nanbeige4.1-3B -4bit --dtype float16 --system "You are concise."
```

## HF CLI runner (raw prompt → completion)

Use `runner.py` when you want the most minimal “type prompt → get completion” flow for HF/Transformers.

Examples:
```bash
python runner.py Nanbeige4.1-3B
python runner.py --config Nanbeige4.1-3B
python runner.py Nanbeige4.1-3B -8bit
python runner.py Nanbeige4.1-3B -4bit
```

## Backend-specific standalone scripts (non-TUI)

These are useful for isolating backend issues or doing quick checks without the unified TUI:

- `alex.py` (GGUF / llama.cpp chat CLI)
  ```bash
  python alex.py /path/to/model.gguf
  ```

- `ollama_chat.py` (Ollama streaming CLI)
  ```bash
  python ollama_chat.py your-ollama-model
  ```

## Legacy / experimental

- `tui_chat.py`: older HF-only Textual TUI script.
  - Prefer `tui.py` unless you’re debugging something specific to that file.

## Model folder scaffolding

To add a new model folder (notes/config/templates/prompts) from `models/_TEMPLATE/`:
```bash
python scripts/model add <model_name> <hf|gguf|ollama> [--id <backend_id>]
```
