# 0004 - Keep backends separated by script

Date: 2026-02-28

## Context
The repo now supports multiple inference backends with different capabilities and model formats:
- Hugging Face checkpoints (`transformers` + `torch`)
- GGUF via `llama-cpp-python`
- Ollama via HTTP API

Trying to force all behaviors into one script would create brittle branching and confusing UX.

## Decision
Use separate entrypoints per backend:
- `chat.py` and `runner.py` for HF text models
- `alex.py` for GGUF
- `ollama_chat.py` for Ollama API

Keep backend-specific logic in the corresponding script rather than building one universal runner.

## Consequences
- Simpler, clearer CLI behavior per backend.
- Faster iteration on backend-specific features (e.g. chat templates vs Ollama think flags).
- Some duplicated utility logic remains acceptable for now to preserve script clarity.

