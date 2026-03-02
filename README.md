# model-runner

Lightweight local model runner scripts for:
- Hugging Face text/chat models (`transformers` + `torch`)
- GGUF models via `llama-cpp-python`
- Ollama models via HTTP API with optional reasoning-filtered output

## User Guide

### 1) Setup

Use the existing virtual env in this repo:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Optional GGUF support:

```bash
pip install llama-cpp-python
```

### 2) Scripts and What They Do

- `runner.py`
  - Simple prompt loop for text generation.
  - Optional token streaming with `--stream`.
  - Supports `-8bit` / `-4bit` (bitsandbytes) on CUDA.
  - Best for non-chat generation tests.

- `chat.py`
  - Chat loop with conversation history.
  - Uses tokenizer native chat template by default.
  - Optional `--prompt-mode plain` for minimal role-formatted prompting (no chat template).
  - Decodes assistant-only new tokens.
  - Optional token streaming with `--stream`.
  - Supports precision selection via `--dtype`.
  - Default `--max-new-tokens` is `2048`.
  - Supports `-8bit` / `-4bit` (CUDA only).

- `tui_chat.py`
  - Legacy Textual TUI for HF chat.
  - Kept as migration reference.

- `tui.py`
  - Unified Textual TUI entrypoint for HF, GGUF, and Ollama.
  - Auto-detects backend from `model_id` or accepts `--backend`.
  - Bottom grey input band with scrollable transcript above.
  - Collapsible streaming thinking section per assistant turn.
  - Hides `<think>` markers while routing inner content to a grey “thinking” panel.
  - Optional `--assume-think` / `--no-assume-think` for models that emit only end-think markers.

- `alex.py`
  - GGUF chat runner via `llama-cpp-python`.
  - Accepts Windows or WSL paths.
  - Best for direct `.gguf` testing without Ollama.

- `ollama_chat.py`
  - Lightweight wrapper around Ollama `/api/chat`.
  - Streams output and can hide reasoning blocks.
  - Auto-detects Ollama host (`OLLAMA_HOST`, localhost, WSL gateway).

### 3) Common Commands

Hugging Face text run:

```bash
python runner.py Nanbeige4.1-3B
python runner.py /home/poop/ml/models/Nanbeige4.1-3B -8bit
```

Hugging Face chat run:

```bash
python chat.py Nanbeige4.1-3B
python chat.py Nanbeige4.1-3B --prompt-mode plain
python chat.py Nanbeige4.1-3B --dtype bfloat16
python chat.py Nanbeige4.1-3B -4bit --dtype float16 --system "You are concise."
```

Config-driven run (recommended):

```bash
python chat.py --config Nanbeige4.1-3B
python chat.py --config models/Nanbeige4.1-3B/hf/config --max-new-tokens 1024
python chat.py --config Nanbeige4.1-3B --stream
python tui.py Nanbeige4.1-3B
python tui.py /mnt/d/models/your-model.gguf
python tui.py /mnt/d/models/your-model.gguf --assume-think
python tui.py ollama:your-ollama-model
python tui.py ollama:your-ollama-model --backend ollama --ollama-think false
python tui_chat.py --config Nanbeige4.1-3B
python tui_chat.py --config Nanbeige4.1-3B --prompt-mode plain
python runner.py --config Nanbeige4.1-3B
```

Template config:

```bash
mkdir -p models/MyModel/hf/config
cp models/_TEMPLATE/hf/config/config.json models/MyModel/hf/config/config.json
```

GGUF run:

```bash
python alex.py "/mnt/d/models/your-model.gguf"
```

Ollama API run:

```bash
python ollama_chat.py "your-ollama-model" --think false
python ollama_chat.py "your-ollama-model" --think false --strict-think-strip
```

### 4) Path Notes

- Bare names like `Nanbeige4.1-3B` auto-resolve to `~/ml/models/<name>` if present.
- Windows paths are accepted by `chat.py`, `runner.py`, and `alex.py` and mapped to WSL style when possible.
- In WSL, Windows drives are typically mounted under `/mnt/<drive-letter>/...`.

### 5) Troubleshooting

- `... is not a local folder and is not a valid model identifier`
  - Use full local path or place model under `~/ml/models/<name>`.

- Tokenizer conversion errors
  - Ensure: `sentencepiece`, `tiktoken`, `protobuf` are installed.

- PersonaPlex model fails in `runner.py`/`chat.py`
  - Expected: PersonaPlex is speech-to-speech, not a text `AutoModelForCausalLM` checkpoint.

- Ollama connection refused from WSL
  - `ollama_chat.py` auto-detects host each run.
  - If needed: `python ollama_chat.py "<model>" --host "http://<windows-ip>:11434"`.

## Configs

- Config files live under `models/<model>/<backend>/config/`.
- Current starter profile: `models/Nanbeige4.1-3B/hf/config/config.json`.
- HF template profile: `models/_TEMPLATE/hf/config/config.json`.
- `--config` lookup supports:
  - direct file path
  - path without `.json`
  - directory containing `config.json` (e.g. `models/<model>/hf/config`)
  - short name resolved under `models/<name>/<backend>/config/config.json` (e.g. `Nanbeige4.1-3B`)

Precedence:

1. CLI flags
2. Config values
3. Script defaults

Selected HF knobs now exposed in CLI/config:

- sampling: `temperature`, `top_p`, `top_k`, `typical_p`, `min_p`
- output mode: `stream`
- routing mode: `assume_think` (TUI)
- length/termination: `max_new_tokens`, `max_time`, `stop_strings`
- repetition/structure: `repetition_penalty`, `no_repeat_ngram_size`
- decoding mode: `num_beams`
- prompt control: `system`, `system_file`, `user_prefix`, `prompt_prefix`
- template control (chat): `chat_template` (`default`, `search`, or template file path)
- chat memory control: `max_context_tokens` (chat only)

## Contributor Guide

### Repo Structure

- `runner.py`: baseline text generation flow.
- `chat.py`: template-aware chat flow.
- `tui.py` + `tui_app/`: unified Textual TUI + backend adapters.
- `alex.py`: GGUF (`llama-cpp-python`) chat backend.
- `ollama_chat.py`: Ollama HTTP streaming backend + filtering.
- `models/`: model-first workspace for per-model config/notes/templates/prompts.
- `docs/`: decisions/specs/audits for repo evolution.
- `requirements.txt`: current `.venv` package lock-style snapshot.
- `models/_shared/`: shared templates and prompt assets (organized by backend).

### Design Constraints

- Keep scripts simple and local-first.
- Prefer explicit, readable CLI behavior over framework complexity.
- Do not mix speech-model pipelines into text runners.
- Keep error messages direct and actionable.

### Development Workflow

1. Edit one script at a time.
2. Run syntax checks:
   - `python -m py_compile runner.py chat.py tui.py tui_chat.py alex.py ollama_chat.py`
3. Validate script help output:
   - `python <script>.py --help`
4. Smoke-test with one known model per backend.

### Extension Ideas

- Add `Modelfile` parser support in `ollama_chat.py` for template/prefix parity.
- Add optional transcript logging (`--log-file`).
- Add streaming token timing metrics for latency debugging.
