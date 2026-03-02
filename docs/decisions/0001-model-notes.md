# 0001 - Capture per-model notes in-repo

Date: 2026-02-28

## Context
Model behavior varies by checkpoint (prompt format, special tokens, tool-call conventions, “thinking” blocks, recommended decoding settings). We need a durable place to record these findings while testing models locally under `/home/poop/ml/models/`.

## Decision
Create a `models/` folder in this repo to store model-specific notes and assets, organized by model then backend:
- `models/<model>/hf/notes/` for Hugging Face / `transformers` checkpoints
- `models/<model>/gguf/notes/` for GGUF models (used with `llama-cpp-python`)
- `models/<model>/ollama/notes/` for Ollama models

Add an initial entry for `Nanbeige4.1-3B` and standardize future HF notes via `models/_TEMPLATE/hf/notes/README.md`.

## Consequences
- We can track “gotchas” and reproducible run settings per model without changing code.
- Notes can include integration guidance (e.g. OpenClaw/tool loops) and “what to try next” before editing scripts.
- The repo remains local-first; model weights stay outside version control.
