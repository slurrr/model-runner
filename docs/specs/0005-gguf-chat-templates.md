# Spec: GGUF chat templates (client-side, no rebuild)

Date: 2026-03-03

## Problem
When a model is run via Ollama, changing the chat template requires rebuilding the model (Modelfile `TEMPLATE`). That makes it slow to iterate on:
- role formatting (System/User/Assistant tokens)
- persona / “agent” constraints
- history sanitation (e.g. dropping `<think>...</think>` from prior assistant messages)

For GGUF (llama.cpp), we should be able to do this **client-side**: keep templates under `models/<model>/gguf/templates/` and apply them at runtime.

## Goal
In the unified TUI (`tui.py`) and GGUF backend (`tui_app/backends/gguf.py`), support:

1) Selecting a GGUF chat template without rebuilding any model.
2) Using a repo-local Jinja chat template file (HF-style `tokenizer.chat_template` Jinja).
3) Optionally selecting a built-in llama.cpp chat format (e.g. `chatml`, `mistral-instruct`) by name.

Non-goals for this spec:
- Tool-call protocols / schema enforcement for GGUF (handled separately).
- “JSON mode” / grammar-constrained decoding (handled separately).

## UX / configuration

### Existing flags to leverage
The unified TUI already has:
- `--prompt-mode {chat,plain}`
- `--chat-template <spec>`

We extend GGUF behavior so these flags apply to `backend=gguf` too.

### `--prompt-mode`
- `chat` (default): use llama.cpp `create_chat_completion(...)`.
- `plain`: bypass chat formatting entirely and use a simple role-labeled prompt with `create_completion(...)`.

### `--chat-template` for GGUF
Interpretation rules for GGUF:

1) If `--chat-template` is empty: use llama.cpp defaults (GGUF metadata / guessed handler).
2) If `--chat-template` points to an existing file:
   - load it as a Jinja chat template (or from JSON if `*.json`)
   - install it as the chat handler for this session
3) Otherwise:
   - treat it as a llama.cpp `chat_format` name (e.g. `chatml`, `mistral-instruct`)

Notes:
- This keeps one knob (`--chat-template`) while still supporting both “file template” and “named format”.
- The resolution order (“file if exists, else name”) is deterministic and easy to explain.

### Config keys
Use the existing JSON config key:
- `"chat_template": ""`

Examples:
- `"chat_template": "models/my-gguf/gguf/templates/current.jinja"`
- `"chat_template": "chatml"`

## Implementation details (code)

### 1) GGUF: apply template/format during session init
File: `tui_app/backends/gguf.py`

Add a helper:
- `resolve_gguf_chat_template_spec(args, config_path) -> tuple[kind, value]`
  - `kind="file"` with absolute resolved path
  - `kind="format"` with name string
  - `kind="auto"` if empty

Reuse the same resolution rules as HF where possible:
- allow relative paths resolved relative to the config file directory

Then, in `create_session(args)` after `llm = Llama(...)`:

- If `prompt_mode == "plain"`:
  - do not install any chat handler (leave defaults)
  - backend will use the existing fallback prompt path
- Else if template spec is `file`:
  - read template text:
    - if `*.json`: parse and pull `chat_template` / `template`
    - else: read raw file text
  - construct llama.cpp Jinja formatter:
    - `from llama_cpp.llama_chat_format import Jinja2ChatFormatter, chat_formatter_to_chat_completion_handler`
    - determine `bos_token` and `eos_token` strings:
      - `bos = llm.detokenize([llm.token_bos()], special=True).decode("utf-8", "ignore")`
      - `eos = llm.detokenize([llm.token_eos()], special=True).decode("utf-8", "ignore")`
      - if either decode yields empty/garbage, allow fallback to `""` (template may not use them)
    - `formatter = Jinja2ChatFormatter(template=text, bos_token=bos, eos_token=eos, add_generation_prompt=True)`
    - `handler = chat_formatter_to_chat_completion_handler(formatter)`
  - install for this model:
    - `llm.chat_handler = handler`
    - optionally set `llm.chat_format = None` to avoid ambiguity
- Else if template spec is `format`:
  - set:
    - `llm.chat_handler = None`
    - `llm.chat_format = <name>`

Print a startup line so it’s obvious what’s in effect:
- `GGUF chat template: auto | format:<name> | file:<path>`

### 2) GGUF: forward common sampling knobs
File: `tui_app/backends/gguf.py`

Currently GGUF only forwards:
- `temperature`, `top_p`, `stop`

Extend to also forward (when set):
- `top_k` (if not `None`)
- `min_p` (if not `None`)
- `typical_p` (if not `None`)
- `repeat_penalty` / `repetition_penalty` (if not default)

Rationale:
- The unified CLI already exposes these knobs.
- llama.cpp supports them in `create_chat_completion(...)`.
- Makes it easier to dial in formatting reliability while iterating on templates.

### 3) Un-ignore `prompt_mode` / `chat_template` for GGUF
File: `tui.py`

Update `_warn_ignored_flags(...)` so for `backend=gguf` we do not warn about:
- `prompt_mode`
- `chat_template`
- `top_k`, `min_p`, `typical_p`, `repetition_penalty` (if/when wired)

## Model folder conventions

For GGUF models in this repo:
- put chat templates under:
  - `models/<model>/gguf/templates/`
- use:
  - `current.jinja` as “active”
  - additional experiments alongside

Recommended pattern:
- keep templates minimal
- if a model emits think tags, sanitize assistant history by stripping pre-`</think>` content before replaying it into the next prompt

## Testing checklist

1) GGUF with embedded chat template (no `--chat-template`): works unchanged.
2) GGUF without embedded chat template: still works via `plain` fallback.
3) `--chat-template models/<m>/gguf/templates/current.jinja`: template is loaded; no rebuild required.
4) `--chat-template chatml`: uses named format handler.
5) Relative `chat_template` path resolution from config file directory works.
6) Start-up prints the chosen template source (auto/format/file).

