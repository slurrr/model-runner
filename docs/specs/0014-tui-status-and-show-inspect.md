#!/usr/bin/env markdown
# Spec: `/status` + a cleaner `/show` (“inspect”) UX for the TUI

Date: 2026-03-08

## Context / problem
We added many slash commands/topics while the repo grew quickly.

Pain points:
- Too many “power user” commands are presented as primary (noisy help).
- Debugging managed backends (notably managed vLLM) is hard because important runtime state/log tails are not visible in the UI.
- Users want a consistent, low-cognitive-load flow:
  - *concise when you want concise*
  - *deep inspection when you need it*

## Goals
- Add a new `/status` command for a concise runtime summary (the “default” debugging view).
- Keep `/show` as the primary deep inspection surface (i.e., “inspect”), but:
  - make it easy to discover topics
  - keep default output minimal
  - provide `--verbose` for the full dump
- Keep backward compatibility:
  - keep existing “shortcut” commands as **hidden aliases** temporarily
  - do not break muscle memory in the short term
- Improve in-TUI debugging for managed backends:
  - show managed vLLM server state (pid/port/url)
  - show recent server log tail on demand

## Non-goals (v1)
- Session save/restore (future).
- Live editing of config files.
- Tool execution loop UI.

## Command set (v1)

### Primary
- `/?` and `/help` (synonyms): show help
- `/status`: concise runtime/session summary
- `/show <topic> [args...]`: deep inspection

### Prompt-related (keep)
- `/system`: show active system prompt and source
- `/prefix`: show prefix/template-related prompt settings

### Utilities (keep)
- `/image ...` (where supported)
- `/clear`
- `/exit`, `/quit`

## `/status` output contract
`/status` should fit on a screen and be stable across backends.

Recommended fields (exact formatting is flexible):

**Identity**
- backend: `hf|gguf|exl2|ollama|openai|vllm|...`
- model: resolved model id / path (as known by the session)
- config: base config path + active profile (if any)

**Run state**
- generating: `true|false`
- follow_output: `true|false`
- pending_images: count

**Gen summary (effective intent)**
- max_new_tokens, temperature, top_p
- include `top_k` only when set (or “default/deferred” if we already track it)

**Backend summary**
- A one-liner derived from `session.describe()` when available.
  - Examples:
    - vLLM: `managed pid=..., base_url=..., model=...`
    - OpenAI attach: `base_url=..., model=...`
    - Ollama: `host=..., model=...`

**Last turn (if available)**
- last elapsed seconds
- last ended_in_think
- lengths: raw/think/answer (already tracked in `TurnRecord`)

Important: `/status` must be safe to use during generation (read-only).

## `/show` redesign (still `/show`, but acts like “inspect”)
`/show` remains the detailed inspector. Topics are stable and additive.

### Default behavior
- `/show` (no args) prints:
  - `Usage: /show <topic> [--verbose]`
  - list of topics (grouped)
  - a couple of examples
  - a pointer to `/status` (“use /status for a concise summary”)

### Common flags (v1)
All `/show` topics should accept:
- `--verbose` (or `-v`): include more fields, expanded dumps, and log tails where relevant
- `--json`: emit JSON for copy/paste (optional per topic; recommended for `args`, `config`, `backend`)

If a topic doesn’t support `--json`, it must say so.

### Per-topic flag support matrix (v1 MUST)
To keep UX consistent and testable, v1 defines the supported flags per topic:

| topic | `--verbose` | `--json` |
|---|---:|---:|
| `status` | ✅ | ✅ |
| `session` | ✅ | ✅ |
| `model` | ✅ | ✅ |
| `prompt` | ✅ | ✅ |
| `gen` | ✅ | ✅ |
| `ui` | ✅ | ✅ |
| `history` | ✅ | ✅ |
| `last` | ✅ | ✅ |
| `files` | ✅ | ✅ |
| `env` | ✅ | ✅ |
| `config` | ✅ | ✅ |
| `backend` | ✅ | ✅ |
| `aliases` | ✅ | ✅ |
| `logs` | ✅ | ❌ |
| `request` | ✅ | ✅ |

Notes:
- `--json` is intentionally disabled for `logs` to avoid “log data as an API” and to keep output human-oriented.

### Topic list (v1)
Keep existing topics but polish output contracts:
- `session`: full session state (superset of `/status`)
- `model`: backend + resolved model id (minimal)
- `prompt`: prompt-related config (system + prefix summary)
- `gen`: “effective generation settings” in sent/deferred/ignored buckets
- `ui`: UI knobs (follow mode, tick, etc.)
- `history`: conversation history summary (counts + last roles)
- `last`: last `TurnRecord` summary
- `files`: resolved file paths (system_file, save_transcript, etc.)
- `env`: env variables that influence behavior (CUDA_VISIBLE_DEVICES, OLLAMA_HOST, etc.)
- `config`: loaded layers + per-key origin map (already implemented)
- `backend`: backend-specific details (session.describe)
- `aliases`: show alias map

#### `/show gen` default output contract (v1 MUST)
To avoid clutter regressions, `/show gen` output is strictly defined:

- Default (`/show gen`):
  - `sent.*` (what is actually sent to the backend)
  - `deferred` (known knobs not sent; backend defaults apply)
  - `ignored` (configured but not supported by this backend)
  - MUST NOT include a full `args.*` dump.

- Verbose (`/show gen --verbose`):
  - includes everything in default output, plus:
  - `args.*` (post-merge config+CLI values)
  - optionally: per-key origin labels if already tracked

### New debugging topics (v1)
Add:
- `logs`: show recent backend logs (when available)
  - Default: a short message (“no logs captured” / “use --verbose”)
  - `--verbose`: show last N lines
  - Optional args:
    - `--n <int>` (default 80)
    - `--filter <substring>` (best-effort, local)

Backend behavior for `logs`:
- v1 MUST support a standardized in-memory ring buffer across all backends.
  - `logs` shows the in-memory ring buffer tail.
  - If the current backend does not emit into the buffer (bug/misconfig), `logs` MUST say so explicitly.

Also show the configured log sink (if any), even if the buffer is empty:
- `log_file` path (if configured)
- `tee` suggestion (when applicable)

Also add:
- `request`: show the last request payload summary (sanitized)
  - Intended for OpenAI-compatible backends and Ollama
  - Must redact secrets (api keys)
  - Default should be short; `--verbose` shows full JSON (truncated to a safe limit)

If we haven’t captured the last request yet, output:
- “No request captured (enable request capture in config)” and explain how.

## Hidden aliases (backward compatibility)
Goal: keep old shortcuts working, but reduce help noise.

### Which aliases to keep (examples)
Keep these as hidden aliases mapping to canonical handlers:
- `/gen` → `/show gen`
- `/args` → `/show args`
- `/config` → `/show config`
- `/backend` → `/show backend`
- `/env` → `/show env`
- `/files` → `/show files`
- `/last` → `/show last`
- `/history` → `/show history`
- `/model` → `/show model`
- `/session` → `/show session`

Help behavior:
- `/help` lists only primary commands by default.
- `/help --all` (or `/help aliases`) shows hidden aliases.

## Implementation notes

### Registry
Keep a single registry for commands. Add metadata:
- `hidden: bool` for help listing
- `read_only: bool` to allow during generation

### `/show` parsing
Continue using `shlex.split`.
`/show <topic> --help` prints that topic’s usage.

### Surfacing managed vLLM logs
All backends should emit log lines to a standardized in-memory ring buffer (size N).
Expose it via `/show logs` and include a short “log_tail available (N lines)” indicator in `/status`.

### Request capture (v1 toggle)
For debugging issues like “decoder prompt cannot be empty” we need to see what the client sent.

Add a config/UI flag:
- `ui.capture_last_request = false` (default)
When enabled:
- store a sanitized copy of the last request payload per backend session (in memory)
- surface via `/show request`

Hard limits + redaction rules (v1 MUST):
- Max stored bytes per captured request: 256 KB.
  - If larger, truncate and include a marker like `"...(truncated)"`.
- Always redact secrets:
  - Authorization headers
  - API keys / bearer tokens
- Always omit or summarize binary/large data:
  - image `data:` URLs MUST NOT be stored verbatim; replace with metadata:
    - `{"type":"image_url","image_url":{"url":"(data-url omitted)","bytes":12345,"mime":"image/png"}}`
  - For local image paths, store only the path basename (or a redacted path if you prefer).

## Acceptance criteria
- `/status` exists and is useful across backends.
- `/show` remains the detailed inspector, with `--verbose`.
- `/show logs` displays vLLM managed log tail without leaving the TUI.
- Hidden aliases work but do not clutter `/help`.
