# model-runner

`model-runner` is a local model lab: a practical workspace for running local models, comparing backends, tuning inference, and testing agent/application behavior without leaving a local-first environment.

This repo spans two overlapping jobs:
- runtime and backend engineering
- AI application, agent, and prompt experimentation

It is intentionally broad. It contains useful tooling, model-specific notes/configs/templates, backend adapters, and a growing observability direction for understanding what models are actually doing at runtime.

## What This Repo Is

This project is for local model research and tinkering:
- run the same model across multiple backends
- compare runtime behavior instead of guessing
- keep model-specific notes, configs, templates, and prompts in-repo
- test tool use, thinking behavior, context management, and backend quirks
- build toward a real local observability and optimization workflow

Supported runtime paths today include:
- Hugging Face / `transformers` + `torch`
- GGUF via `llama-cpp-python`
- Ollama via HTTP
- EXL2 via ExLlamaV2
- OpenAI-compatible servers
- managed vLLM

## Current State

This repo is already useful, but uneven.

What exists now:
- a multi-backend Textual TUI for local interaction and debugging
- simple CLI entrypoints for isolated backend checks
- model-first config/notes/templates layout under `models/`
- a lot of backend standardization work around prompts, knobs, history, logging, token accounting, and tool calls

What is still rough:
- the TUI is a workable control surface, but not a polished observability product
- backend comparison is possible, but not yet clean or visual enough for serious runtime learning
- some runtime truths are still too hard to surface quickly

If you want to just chat with models, the current tooling works.
If you want to learn how backends really behave, optimize them, and compare them with confidence, the repo still needs a stronger observability layer.

## Direction

The direction is a **local model lab**, not “just a TUI.”

That means:
- keep local-first model execution
- preserve backend diversity instead of collapsing everything into one engine
- make runtime behavior visible and measurable
- make tuning and comparison enjoyable, not forensic
- move toward a browser-based observability and control surface for metrics, backend state, cache growth, memory behavior, and experiment comparison

The existing observability direction is captured in:
- [docs/observability_dashboard](/home/poop/ml/model-runner/docs/observability_dashboard)

The current TUI should be thought of as:
- a current interaction surface
- a useful backend bring-up tool
- not the final home for optimization, inspection, and comparison UX

## Entry Points

### Current control surface: unified TUI

Use `tui.py` when you want a terminal UI for chat, streaming, thinking display, slash-command inspection, or fast backend switching.

Optional: install the console entrypoint:
```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Examples:
```bash
tui Nanbeige4.1-3B
python tui.py Nanbeige4.1-3B
python tui.py --config Qwen3.5-9B --backend vllm
```

### Backend bring-up: managed vLLM helpers

Use the helper commands when you want repo-configured managed vLLM without entering the TUI:

```bash
./vllm-up --config Qwen3.5-9B
./vllm-up --config Qwen3.5-9B --bg
./vllm-down --config Qwen3.5-9B
```

### Isolation/debug CLIs

Use these when you want simpler, backend-specific loops:

- `runner.py`
  - minimal HF prompt -> completion
- `chat.py`
  - template-aware HF chat loop
- `alex.py`
  - GGUF chat via `llama-cpp-python`
- `ollama_chat.py`
  - Ollama streaming chat loop
- `tui_chat.py`
  - older HF-only Textual TUI kept as reference/debug aid

More detail lives in:
- [docs/entrypoints.md](/home/poop/ml/model-runner/docs/entrypoints.md)

## Model-First Workspace

The repo is organized around models first, then backends:

- `models/<model>/<backend>/notes/`
- `models/<model>/<backend>/config/`
- `models/<model>/<backend>/templates/`
- `models/<model>/<backend>/prompts/`

That structure exists so research stays grounded:
- notes next to configs
- templates next to the model they affect
- backend variants for the same model stay comparable

To scaffold a new model folder:
```bash
python scripts/model add <model_name> <hf|gguf|ollama|exl2> [--id <backend_id>]
```

## Setup

Use the repo venv:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Optional backend extras:
- GGUF: `pip install llama-cpp-python`
- EXL2: see [docs/exl2_setup.md](/home/poop/ml/model-runner/docs/exl2_setup.md)

## Common Commands

Unified TUI:
```bash
tui Nanbeige4.1-3B
tui /mnt/d/models/your-model.gguf
tui ollama:your-ollama-model
tui --backend exl2 /path/to/exl2_model_dir
```

HF text run:
```bash
python runner.py Nanbeige4.1-3B
python runner.py /home/poop/ml/models/Nanbeige4.1-3B -8bit
```

HF chat run:
```bash
python chat.py Nanbeige4.1-3B
python chat.py Nanbeige4.1-3B --prompt-mode plain
python chat.py Nanbeige4.1-3B --dtype bfloat16
python chat.py Nanbeige4.1-3B -4bit --dtype float16 --system "You are concise."
```

Config-driven run:
```bash
python chat.py --config Nanbeige4.1-3B
python chat.py --config models/Nanbeige4.1-3B/hf/config --max-new-tokens 1024
python chat.py --config Nanbeige4.1-3B --stream
python runner.py --config Nanbeige4.1-3B
python tui.py --config Nanbeige4.1-3B --backend hf
```

GGUF run:
```bash
python alex.py "/mnt/d/models/your-model.gguf"
```

Ollama run:
```bash
python ollama_chat.py "your-ollama-model" --think false
python ollama_chat.py "your-ollama-model" --think false --strict-think-strip
```

## Configs

- Config files live under `models/<model>/<backend>/config/`
- `--config` lookup supports:
  - direct file path
  - path without extension
  - directory containing config
  - short name resolved under `models/<name>/<backend>/config/`

Precedence:
1. CLI flags
2. Config values
3. Script defaults

## Troubleshooting

- `... is not a local folder and is not a valid model identifier`
  - Use a full local path or place the model under `~/ml/models/<name>`

- Tokenizer conversion errors
  - Ensure `sentencepiece`, `tiktoken`, and `protobuf` are installed

- PersonaPlex model fails in `runner.py`/`chat.py`
  - Expected: it is a speech-to-speech checkpoint, not a text causal LM

- Ollama connection refused from WSL
  - `ollama_chat.py` auto-detects host each run
  - If needed: `python ollama_chat.py "<model>" --host "http://<windows-ip>:11434"`

## Contributor Notes

This repo is not trying to be a minimal polished app. It is trying to become a useful local model research environment.

From a contributor perspective, the repo operates on two different kinds of documents:
- **Decision records** capture contracts and invariants the repo has discovered or chosen.
  - They explain what must stay true across refactors, standardization passes, and new backend work.
  - Treat them as the repo's operating agreements, not as disposable notes.
- **Specs** describe concrete implementation passes.
  - They are instructions for how to execute a change or feature once the direction is understood.
  - They may evolve, but they should not silently violate existing decision records.

In practice:
- consult decision records first when you need to understand what behavior is supposed to remain invariant
- use specs to drive the implementation work for a specific pass
- if implementation reveals a new invariant, ambiguity, or repo-wide rule, capture that in a decision record instead of leaving it trapped in code or chat history

When making changes:
- prefer backend truth over inferred summaries
- preserve local-first workflows
- keep models comparable across backends where practical
- document model/runtime quirks in-repo
- treat observability as a first-class feature, not a nice-to-have
- distinguish clearly between:
  - requested/configured behavior
  - effective/runtime behavior
  - unknown/unavailable runtime truth
- avoid hiding backend-specific realities behind fake uniformity; standardize the surface, not the facts

The current TUI is still worth improving, but new work should not assume terminal UX is the final destination for runtime debugging and optimization.
