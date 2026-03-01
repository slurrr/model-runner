# AGENTS

## Purpose
This repository hosts lightweight local model runners for multiple backends:
- Hugging Face causal language models (`transformers` + `torch`)
- GGUF models (`llama-cpp-python`)
- Ollama API models (streaming wrapper)

## Scope
- Keep the project small and practical.
- Prioritize simple CLI flows for loading models and chatting.
- Prefer local-first execution with PyTorch + Transformers.

## Current Entry Points
- `runner.py`: minimal interactive text generation runner.
- `chat.py`: template-aware chat loop with conversation history.
- `alex.py`: GGUF chat runner via `llama-cpp-python`.
- `ollama_chat.py`: Ollama API wrapper with optional reasoning filtering.

## Conventions
- Default to straightforward Python scripts over heavy framework structure.
- Keep dependencies minimal and rely on the existing `.venv`.
- Favor clear failure messages when CUDA/model loading is unavailable.
- Keep backend-specific logic separated instead of overloading one script.

## Usage
Run:

```bash
python runner.py <hf_model_or_local_path>
python chat.py <hf_model_or_local_path>
python alex.py <path_to_model.gguf>
python ollama_chat.py <ollama_model_name>
```

Type prompts at `>` and type `exit` or `quit` to end.
