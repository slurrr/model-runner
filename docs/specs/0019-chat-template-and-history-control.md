#!/usr/bin/env markdown
# Spec: Chat template + history control (best effort, repo-wide)

Date: 2026-03-09

Status: Draft

Related audit:
- `docs/audits/0003-standardization.md`

Related umbrella contract:
- `docs/specs/0015-backend-standardization-contracts.md`

## Context / problem
Model behavior varies dramatically based on:
- chat template formatting
- whether “thinking” content is included in conversation history

This repo’s direction is: **model-first, repo-controlled**, and “best-effort” consistency across engines/backends.

However, not every backend gives us the same control surface:
- HF / GGUF / EXL2: client-side prompt rendering is under repo control.
- vLLM managed: server launch is under repo control; template control depends on vLLM flags/version.
- Ollama `/api/chat`: template is baked into the model/build and not worth trying to override in this repo.

## Goals
- Standardize a `chat_template` knob across all backends where it’s meaningful.
- Standardize history sanitation so we can avoid feeding raw `<think>...</think>` back into the model context (when desired).
- Keep behavior explicit per backend (no silent “we tried”).

## Non-goals
- Forcing Ollama `/api/chat` to use repo templates.
- Tool-call schema templating in this spec (separate work).

## Standard config knobs (required)
Repo-wide knob (always present):
- `chat_template` (string):
  - file path (repo-relative or config-relative), or
  - empty to mean “use backend/model default”, or
  - ignored when the backend cannot honor it (see capability reporting).

Backends that can honor templates (MUST):
- If `template_control_level` is `local_template` or `managed_server_template`, the backend MUST attempt to apply `chat_template` when set.
Backends that cannot honor templates (MUST):
- If `template_control_level` is `server_owned_template`, the backend MUST report the template override as ignored and MUST NOT claim it took effect.

All backends MUST support:
- `history_strip_think` (bool, default: false):
  - If true: when preparing the messages that will be re-fed on the next turn, remove any `<think>...</think>` spans.
  - If `assume_think=true`, apply the same policy to “assumed-think” spans (i.e., content routed to the think channel even without a start marker).
  - Display in the UI is unchanged; this only affects future turns.

Notes:
- `history_strip_think` is independent of `show_thinking`.

## Capability reporting (required)
Every backend session MUST expose (via `describe()` or `/show backend`):
- `template_control_level`:
  - `local_template` (HF / GGUF / EXL2)
  - `managed_server_template` (vLLM managed mode)
  - `server_owned_template` (Ollama `/api/chat`, external OpenAI server attach)

Every backend session SHOULD also expose:
- `chat_template_requested: str` (raw config value; resolved path only when local file-based and safe to disclose, otherwise keep raw spec)
- `chat_template_applied: bool`
- `chat_template_reason: str` (e.g. `applied`, `ignored_server_owned`, `unsupported_flag`, `empty_default`)

This is for UX and debugging (so “why isn’t template change working?” is obvious).

## Backend specifics
### HF
- `chat_template` overrides `tokenizer.chat_template` (and processor tokenizer if present).
- `history_strip_think` implemented by sanitizing assistant `content` in `trimmed_messages` (the messages that will be re-fed next turn).

### GGUF
- `chat_template` supports:
  - file-based Jinja template (preferred for repo-controlled templates)
  - llama.cpp built-in chat formats (when selected explicitly)
- `history_strip_think` applied to the messages that will be re-fed next turn (trimmed/sanitized history).

### EXL2
- `chat_template` controls local Jinja rendering.
- `history_strip_think` applied to the messages that will be re-fed next turn (trimmed/sanitized history).

### vLLM managed
- `chat_template` best-effort:
  - attempt to pass template flag(s) at server launch when configured
  - if the flags are not supported at runtime, report `chat_template_applied=false` with `chat_template_reason=unsupported_flag`
- `history_strip_think` is always applied client-side before sending `messages` payload (when enabled).

### OpenAI (external attach)
- `chat_template` is ignored (report explicitly as ignored in `/show gen` / `/show backend`).
- `history_strip_think` still applies client-side (we control message history).

### Ollama
- `chat_template` is ignored (template is baked into the build).
- `history_strip_think` applies to future requests only (it cannot retroactively change a completed turn).

## Think-only turns under `history_strip_think=true` (MUST)
Some turns may end with thinking only (no final answer), especially with `assume_think=true` or when the model fails to emit an end marker.

Rule:
- If `history_strip_think=true` and the assistant content becomes empty/whitespace after stripping, the backend MUST omit that assistant message
  from the `trimmed_messages` that will be re-fed next turn.
  - This prevents injecting an empty assistant turn into the next request.
  - The UI transcript still retains the full `TurnRecord` for inspection.

If a backend wants additional visibility, it MAY add a `notes[]` entry in the generation knob report (spec 0018) such as:
- `history_strip_think: dropped assistant message (think-only)`

## Testing checklist
- Switch `chat_template` file and confirm prompt formatting changes (HF/GGUF/EXL2).
- Enable `history_strip_think` with a reasoning model:
  - verify the UI still shows thinking
  - verify next-turn prompt sent to backend does not include `<think>` spans.
- vLLM managed:
  - verify template support is reported correctly (applied vs ignored).
