# Spec: Unified TUI entrypoint (HF + GGUF + Ollama)

Date: 2026-03-02

## Context
We have a working Textual TUI (`tui_chat.py`) that targets Hugging Face (Transformers) chat models. This repo also supports:
- GGUF models via `llama-cpp-python` (`alex.py`)
- Ollama API models (`ollama_chat.py`)

Goal: a single entrypoint `python tui.py <model>` that auto-detects which backend to use and provides the same TUI UX (streaming, collapsible thinking, scroll/follow, transcript saving).

## Goals
- One unified entrypoint: `python tui.py <model_or_path_or_ollama_name>`
- Backend auto-detection with an explicit override (`--backend`) when ambiguous.
- Keep the current TUI look/feel and interaction model:
  - fixed grey input band
  - transcript scroll + follow mode
  - “thinking” in grey, answer in white
  - click / keybind to expand “thinking”
- Maintain backend isolation (no giant “if hf/gguf/ollama” spaghetti inside UI widgets).
- Preserve “local-first” operation; optional deps remain optional (GGUF backend only requires `llama-cpp-python` when used).

## Non-goals (for this iteration)
- Agent/tool execution UI (we may render tool calls later, but won’t execute them).
- A fully general “any modality” runner (vision/audio/speech) inside the TUI.
- Perfect tool-call compatibility across all models/vendors.

## Naming / import safety
Avoid naming conflicts where a top-level `tui.py` script shadows a `tui/` package (e.g. `from tui.app import ...` can import the script instead of the package depending on `sys.path`/cwd).

Decision for this spec:
- Keep the **entrypoint** as `tui.py` (nice CLI).
- Name the **package** `tui_app/` (so imports look like `from tui_app.app import ...`).

## Proposed file layout
- `tui.py` (new): unified CLI + backend selection + launches the Textual app
- `tui_app/` (new package directory)
  - `tui_app/app.py`: Textual UI (transcript widgets, input band, scrolling/follow, per-turn rendering)
  - `tui_app/events.py`: event types / protocol (dataclasses)
  - `tui_app/think_router.py`: model-output routing to think vs answer (tag parser + optional adapters)
  - `tui_app/backends/`
    - `hf.py`: Transformers backend (model/tokenizer load, chat_template, generate/stream)
    - `gguf.py`: llama.cpp backend (load GGUF, create_chat_completion streaming)
    - `ollama.py`: Ollama backend (HTTP streaming, host detection)

Notes:
- `tui_chat.py` can be kept temporarily as a reference / migration aid, then removed once `tui.py` is complete.
- Keep existing backends’ standalone scripts (`chat.py`, `runner.py`, `alex.py`, `ollama_chat.py`) unchanged.

## Backend selection / auto-detect

### Selection precedence
1. `--backend {hf,gguf,ollama}` if provided.
2. If `model` starts with `ollama:` → Ollama backend, model name = suffix.
3. If `model` is a path ending in `.gguf` (after WSL path normalization) → GGUF backend.
4. If `model` is an existing local directory → HF backend (local HF-format checkpoint).
5. Otherwise → HF backend (treat as HF Hub id). (If running offline, this will fail fast with a clear error.)

### Why require `ollama:` prefix (by default)
Ollama model names can look like HF ids (e.g. `llama3`) and can’t be reliably detected without calling the Ollama API. A scheme prefix keeps detection deterministic and avoids surprising “wrong backend” picks.

Optional later enhancement:
- If `--backend` omitted and `OLLAMA_HOST` reachable and `model` is not a path, probe `/api/tags` for an exact match, then choose Ollama.

### Config stem mapping for Ollama names
Ollama model names can contain characters that are awkward as folder names (notably `:` as in `foo:latest`).

Decision for this spec:
- Keep config folders **filesystem-safe** by using a sanitized “stem”.
- Store the exact backend model identifier in config (e.g. `"model": "foo:latest"`), and use the sanitized stem only for the folder name.

Proposed sanitizer:
- Replace `/` and `:` with `__` (double underscore).
  - Example: `ollama:foo:latest` → backend model name `foo:latest`, config stem `foo__latest`
  - Example: `ollama:org/model:tag` → `org__model__tag`

## Unified event protocol (UI-facing)
UI should consume backend output as a stream of small “events” rather than backend-specific objects.

### Event types
- `TurnStart(turn_id: int)`
- `ThinkDelta(turn_id: int, text: str)` (grey)
- `AnswerDelta(turn_id: int, text: str)` (white)
- `Meta(turn_id: int, key: str, value: Any)` (token counts, timings, model info)
- `Error(turn_id: int, message: str)`
- `Finish(turn_id: int, record: TurnRecord)`

### Event ordering guarantees
- For a given `turn_id`, the backend MUST emit:
  - exactly one `TurnStart` first
  - zero or more `ThinkDelta` / `AnswerDelta` / `Meta` in any order after start
  - at most one terminal event: `Finish` OR `Error`
- UI may assume deltas belong to the most recent active turn, but should still key by `turn_id`.

### TurnRecord (for saving transcript/debug)
- `raw`: full raw model text if available (may be empty for backends that don’t expose it)
- `think`: extracted think text
- `answer`: extracted answer text
- `ended_in_think`: bool
- `backend`: `"hf" | "gguf" | "ollama"`
- `model_id`: resolved model identifier (path / ollama name / HF id)
- `gen`: generation settings snapshot (max tokens, temp, top-p, etc.)
- `timing`: start/end timestamps + elapsed seconds (if available)

The Textual app stores a per-turn `TurnRecord` and may optionally append JSONL to `--save-transcript`.

## Thinking / answer routing
We want consistent behavior across backends:
- Hide marker tokens (`<think>...</think>` and known alternates).
- Render the content *between* markers as “thinking” (grey), collapsible.
- Render outside markers as “answer” (white).

### Router interface
`ThinkRouter` is a small incremental parser:
- `feed(text_chunk) -> list[(channel, text)]` where `channel in {"think","answer"}`
- `flush() -> list[(channel, text)]`
- exposes `mode` (`"think"`/`"answer"`) to compute `ended_in_think`.

### Backend-specific notes
- HF: current `StreamingThinkParser` approach works; token-id assistance is optional.
- GGUF: llama.cpp streaming returns deltas; feed those into the same router.
- Ollama: prefer routing:
  - if API provides separate `message.thinking`, emit that as `ThinkDelta` directly,
  - otherwise run router on `message.content`.

## Configuration
Unify config semantics while allowing backend-specific knobs.

### `--config` resolution
Use existing `config_utils.load_json_config(config_arg, backend=...)` with backend determined by selection rules above.

### Recommended config locations (model-first)
- HF: `models/<ModelName>/hf/config/config.json`
- GGUF: `models/<ModelName>/gguf/config/config.json`
- Ollama: `models/<ModelName>/ollama/config/config.json`

### Shared settings (examples)
- `max_new_tokens`, `temperature`, `top_p`, `top_k`, `stop_strings`
- `show_thinking`, `no_animate_thinking`
- `save_transcript`
- `system`, `system_file`, `user_prefix`

### Backend-specific settings (examples)
- HF: `dtype`, `use_4bit`, `use_8bit`, `chat_template`, `max_context_tokens`
- GGUF: `n_ctx`, `n_gpu_layers`, `max_tokens` (map from unified `max_new_tokens`), `repeat_penalty` mapping, etc.
- Ollama: `host`, `timeout`, `think` (Ollama-specific “think” toggle), etc.

## CLI surface (unified)
`python tui.py <model> [--backend ...] [--config ...] [common flags...] [backend flags...]`

### Common flags (stable across backends)
- `--system`, `--system-file`, `--user-prefix`
- `--max-new-tokens`
- `--temperature`, `--top-p`, `--top-k`
- `--stop-strings ...`
- `--show-thinking`, `--no-animate-thinking`
- `--save-transcript <path>`

### Backend-specific flags (namespaced or flat)
Option A (flat, simplest, matches current scripts):
- expose a superset of flags; backend ignores what it doesn’t use.

Option B (namespaced, clearer long-term):
- `--hf-dtype`, `--hf-4bit`, ...
- `--gguf-n-ctx`, `--gguf-n-gpu-layers`, ...
- `--ollama-host`, `--ollama-timeout`, ...

Decision for MVP:
- Use **flat flags** for parity with existing scripts; introduce namespacing only if collisions become annoying.
- Print a one-time startup summary of **ignored flags** for the selected backend (to avoid “silent no-op” confusion).

## Implementation plan (incremental)
1. Extract UI portion of `tui_chat.py` into `tui/app.py` so it only knows how to render `ThinkDelta`/`AnswerDelta` and manage follow/scroll.
2. Implement `tui/events.py` dataclasses and a small event queue bridge (thread-safe).
3. Implement `tui/backends/hf.py` by moving the HF model load + generation logic out of `tui_chat.py`.
4. Implement `tui/backends/ollama.py` by adapting `ollama_chat.py` streaming loop to emit events.
5. Implement `tui/backends/gguf.py` by adapting `alex.py` to:
   - use `create_chat_completion(..., stream=True)` when available,
   - fall back to completion prompting if not,
   - emit deltas as events.
6. Add `tui.py` entrypoint with auto-detect + config loading + backend wiring.
7. Keep `tui_chat.py` temporarily; mark as legacy in docs; remove after `tui.py` is stable.

## Testing / validation checklist
- HF: local checkpoint with chat_template; verify thinking collapse/expand and transcript saving.
- GGUF: `.gguf` with known chat template; verify streaming and scroll/follow.
- Ollama: `ollama:<model>` with streaming on; verify thinking handling if `message.thinking` present.
- Terminal behavior: validate keybindings that terminals reliably forward (consider adding `Ctrl+U`/`Ctrl+D` as scroll up/down in addition to PageUp/PageDown).
- Optional deps: verify `python tui.py <hf_model>` starts and runs even if `llama-cpp-python` is not installed.

## Stop / cancellation behavior
MVP scope clarification:
- `/stop` / cancellation is out-of-scope for the first unified TUI unless explicitly added later.
- If implemented later, define backend-specific behavior:
  - HF: signal / stopping criteria / generation thread cancellation
  - GGUF: stop the streaming iterator
  - Ollama: abort HTTP request

## Windows path normalization
If `<model>` is a Windows-style drive path (e.g. `C:\\models\\foo.gguf`) running under WSL, normalize to `/mnt/c/models/foo.gguf` before backend detection (reuse the same logic as `alex.py` / HF runners).

## Backend-specific notes / gotchas
- Meta fields are optional:
  - Timing is always possible (wall-clock), but token counts may be approximate for GGUF/Ollama unless instrumented.
- Stop strings differ by backend:
  - HF: `stop_strings` uses Transformers stopping support (string-based stopping in `generate()`).
  - GGUF: llama.cpp stop is typically a list of strings; semantics may differ (byte-level vs token-level boundaries).
  - Ollama: stop support depends on Ollama API options and the model; document behavior and default expectations.
- Lazy imports:
  - `llama_cpp` must only be imported inside the GGUF backend code path.
  - Ollama HTTP/client code should only be imported/executed inside the Ollama backend code path.
- Ollama thinking routing:
  - If the API provides separate `message.thinking`, emit it as thinking and do not also run the generic `<think>` router on the same content.
