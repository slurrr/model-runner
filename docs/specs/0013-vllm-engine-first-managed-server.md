#!/usr/bin/env markdown
# Spec: vLLM as a first-class engine backend (managed server)

Date: 2026-03-08

## Context / problem
Today the unified TUI treats most backends as “engines”:
- `hf` (Transformers, in-process)
- `gguf` (llama.cpp via `llama-cpp-python`, in-process)
- `exl2` (ExLlamaV2, in-process)
- `ollama` (Ollama server, out-of-process)

But the vLLM work was introduced as an `openai` backend (OpenAI-compatible HTTP protocol). That makes the repo inconsistent:
- In the repo, “backend” usually means **what runs the model** (engine).
- `openai` is instead **how we talk to a server** (transport/protocol).

This spec makes vLLM “first-class” in the same way as EXL2: you choose vLLM as an engine, the TUI starts/stops it, and the transport used to speak to it is an implementation detail.

## Goals
- Add a **`vllm` engine backend** that:
  - starts a vLLM OpenAI-compatible server process on launch
  - waits for readiness
  - streams chat via the existing OpenAI-compatible client logic
  - stops the server on clean exit
- Make user workflow consistent:
  - `tui Qwen3.5-9B vllm`
  - model config lives at `models/<model>/vllm/...`
- Keep future flexibility:
  - v2 can add “persist server” mode
  - later engines that expose OpenAI-compatible APIs (e.g. “tabby server”) can reuse the same transport module without becoming “transport-named backends”.

## Non-goals (v1)
- Persistable server lifecycle (leave vLLM running after exit). Design so v2 can bolt this on.
- Supporting GGUF/EXL2 through vLLM (not a vLLM target).
- Full tool execution loop in TUI (still “surface only”).

## Design overview (2-layer internal model)
Keep the user-facing choice as **engine backends**.

Internally:
- “Engine backend” = chooses how a model runs.
- “Transport client” = how we talk to an engine when it is out-of-process.

For vLLM:
- engine backend: `vllm`
- transport client: `openai_http` (OpenAI-compatible chat-completions streaming)

Important: `openai_http` is a reusable module, not a user-facing engine choice in the normal local workflow.

## CLI / UX
### Backend selection
Add `vllm` as a backend choice:
- positional: `tui <model> vllm`
- flag: `tui <model> --backend vllm`

Deprecation stance (docs only, v1):
- Prefer `vllm` for local vLLM usage.
- Keep the existing `openai` backend as the “external OpenAI-compatible server attach” backend (supported in v1; not deprecated).

### Profiles
Profiles remain the mechanism for “combos”:
- `tui <model> vllm --profile maxctx`
- `tui <model> vllm --profile tp2`

Profiles should be engine-scoped and narrow: only override vLLM-related knobs for that model.

## Repo layout
Add model-first layout for vLLM:
- `models/<model>/vllm/config/default.toml`
- `models/<model>/vllm/config/profiles/<name>.toml`
- `models/<model>/vllm/notes/README.md`
- `models/<model>/vllm/templates/` (optional; vLLM uses tokenizer templates indirectly, but we may still keep reference templates here)
- `models/<model>/vllm/prompts/` (ignored by git, may contain secrets)

Also add:
- `models/_TEMPLATE/vllm/...`

## Config shape (TOML)
Constraint: current config loader flattens only `backend.<selected_backend>.*`.
Therefore, vLLM config must include **both** server-lifecycle knobs and OpenAI-HTTP transport knobs under `backend.vllm`.

Recommended template (`models/_TEMPLATE/vllm/config/default.toml`):
```toml
[model]
id = ""                 # local path to HF model folder OR a served model name override (see below)

[gen]
stream = true
max_new_tokens = 1024
temperature = 0.7
top_p = 1.0
stop_strings = []

[prompt]
system = ""
system_file = ""
user_prefix = ""

[ui]
show_thinking = true
assume_think = false

[backend.vllm]
# Lifecycle
mode = "managed"        # "managed" (v1), future: "external", "managed_persist"

# Where to bind the server (managed mode)
host = "127.0.0.1"
port = 8000             # if 0, auto-select a free port (recommended to avoid collisions)

# How to start vLLM
cmd = "vllm"            # or python module invocation; keep configurable
extra_args = []         # list of extra CLI args (strings), appended verbatim

# OpenAI-compatible client transport (used to talk to the managed server)
base_url = ""           # if empty in managed mode, derive from host/port as http://host:port/v1
api_key = ""            # optional; prefer env var if set
timeout_s = 600

# vLLM common knobs (examples; keep modest in v1)
served_model_name = ""  # if set, pass --served-model-name (controls `model` string clients must send)
tensor_parallel_size = 1
gpu_memory_utilization = 0.90
max_model_len = 0       # 0 means “let vLLM decide”; otherwise pass explicit
```

### Model id semantics
We need a clear rule so “run a model” is consistent with other engines:
- In managed mode:
  - If `model.id` is a local directory, use it as the vLLM `--model` path.
  - `served_model_name` controls the OpenAI-visible model name (and the `model` field the client sends).
- In external mode (future):
  - `model.id` is a server-side model name (string), and we do not start a server.

## Implementation plan (code-level)
### 1) Factor OpenAI-compatible HTTP logic into a reusable module
Refactor existing `tui_app/backends/openai.py` into something like:
- `tui_app/transports/openai_http.py` (preferred), or
- keep file but make it clearly “transport” with no engine-specific naming.

It should accept:
- `base_url`, `api_key`, `timeout_s`
- a resolved `model` string
- messages + gen knobs
and return the same streaming event types used today.

### 2) Add a `vllm` backend that manages server lifecycle
Add:
- `tui_app/backends/vllm.py` (engine backend)
Responsibilities:
- Build server command from config:
  - `cmd` + `serve`/module invocation + `--host/--port`
  - `--model <local_path>`
  - optional: `--served-model-name`
  - optional knobs (tp, gpu mem util, max_model_len, etc)
- Spawn with `subprocess.Popen(argv, shell=False, ...)` (MUST be argv-list; no shell string execution).
  - If `cmd` is a string, split using `shlex.split(cmd)` into argv.
  - If we later support `cmd_argv` (list), prefer that as the most deterministic.
- Port collision handling (MUST):
  - If `port == 0`, select a free port before spawning (e.g. bind a local socket to port 0, read assigned port, close, then launch vLLM with that port).
  - If `port != 0` and spawn fails due to bind error, fail fast with a clear message.
    - Include the attempted host/port and suggest a “find free port” command (e.g. `ss -ltnp | rg :8000`).
- Wait for readiness (MUST):
  - Normalize to a canonical OpenAI base URL ending in `/v1`.
  - Probe readiness by polling `GET <base_url>/models` (i.e. `/v1/models`) until success or timeout.
- fail with clear error output if server does not come up
- Once ready: create an OpenAI HTTP transport session and stream tokens.
- On app exit: terminate the process group and wait with a short timeout.
  - Shutdown escalation (MUST): TERM → wait short timeout → KILL if still alive.
  - Lifecycle scope (MUST): start server once per TUI session, not per turn.
  - Abnormal exit behavior (MUST): ensure shutdown occurs on `KeyboardInterrupt` and normal TUI exit; best-effort on signals (wrap main run loop with `try/finally`).

Startup failure diagnostics (MUST):
- Capture stdout/stderr to a ring buffer (last N lines).
- On readiness timeout or non-zero exit:
  - show exact launch argv
  - show last N lines from captured logs

### 3) Wire into `tui.py`
- Add `vllm` to:
  - backend choices (`backend_hint` and `--backend`)
  - default config lookup
  - `create_session` dispatch table

### 4) Templates / onboarding
- Add `models/_TEMPLATE/vllm/` and ensure `scripts/model add ... vllm` works (copy only the vllm subtree).
- Decide where to import model cards for vllm (usually same as HF because the served model is an HF folder).

## Deterministic “model id” contract (managed mode) (MUST)
In managed mode we are starting a vLLM server intended to serve *one* model. The client MUST always send one unambiguous `model` string.

Rules:
- If `backend.vllm.served_model_name` is set:
  - pass it to vLLM (`--served-model-name ...`)
  - client MUST send that same name in requests
- Else:
  - after readiness, call `GET <base_url>/models`
  - if exactly one model is listed, client MUST use that id
  - otherwise fail with a clear message telling the user to set `backend.vllm.served_model_name`

This avoids overloading `[model].id` with “local path” vs “server-visible model string”.

## Config plumbing / key mapping plan (MUST)
Because config flattening is backend-scoped, vLLM needs explicit flatten + argparse wiring so config values are not silently ignored.

Flatten mapping (TOML → flat keys):
- `[backend.vllm] host` → `vllm_host`
- `[backend.vllm] port` → `vllm_port`
- `[backend.vllm] mode` → `vllm_mode`
- `[backend.vllm] cmd` → `vllm_cmd`
- `[backend.vllm] extra_args` → `vllm_extra_args`
- `[backend.vllm] served_model_name` → `vllm_served_model_name`
- `[backend.vllm] tensor_parallel_size` → `vllm_tensor_parallel_size`
- `[backend.vllm] gpu_memory_utilization` → `vllm_gpu_memory_utilization`
- `[backend.vllm] max_model_len` → `vllm_max_model_len`

Transport keys under vLLM (vLLM uses OpenAI-compatible HTTP under the hood):
- `[backend.vllm] base_url` → `vllm_base_url` (optional; derived from host/port when empty)
- `[backend.vllm] api_key` → `vllm_api_key`
- `[backend.vllm] timeout_s` → `vllm_timeout_s`

Argparse + CLI flags should mirror these names (with `--vllm-*` flags) and continue to support `/show gen` showing sent/deferred/ignored.

## Knob parity for vLLM (MUST)
User expectation: vLLM should not be a “minimal backend”. If the repo already exposes a knob for other engines, vLLM should:
- send it when vLLM supports it, or
- clearly mark it as **ignored** (not silently dropped).

### Chat-completions request knobs to support (v1 MUST)
Send these in the `/v1/chat/completions` JSON body when set:

OpenAI-standard (expected to work on vLLM and most compatible servers):
- `max_new_tokens` → `max_tokens`
- `temperature` → `temperature`
- `top_p` → `top_p`
- `stop_strings` → `stop`
- `presence_penalty` → `presence_penalty`
- `frequency_penalty` → `frequency_penalty`

vLLM-supported extras (supported by vLLM Chat API as “extra parameters”; safe to use for the **vllm engine backend**):
- `top_k` → `top_k`
- `min_p` → `min_p`
- `repetition_penalty` → `repetition_penalty`
- `stop_token_ids` (new) → `stop_token_ids`
- `ignore_eos` (new) → `ignore_eos`
- `min_tokens` (new) → `min_tokens`
- `seed` → `seed`
- `best_of` (new) → `best_of`
- `use_beam_search` (new) → `use_beam_search`
- `length_penalty` (new) → `length_penalty`
- `include_stop_str_in_output` (new) → `include_stop_str_in_output`
- `skip_special_tokens` (new) → `skip_special_tokens`
- `spaces_between_special_tokens` (new) → `spaces_between_special_tokens`
- `truncate_prompt_tokens` (new) → `truncate_prompt_tokens`
- `allowed_token_ids` (new) → `allowed_token_ids`
- `prompt_logprobs` (new) → `prompt_logprobs`

Notes:
- vLLM’s docs explicitly mention passing non-OpenAI parameters (e.g. `top_k`) as “extra parameters” / `extra_body` in the OpenAI client; for direct HTTP, they are simply included in the request JSON body.
- Some OpenAI-compatible providers reject unknown fields; this is a key reason vLLM should be its own engine backend (so we can safely send vLLM extras there without breaking generic “openai attach” mode).

### Knobs that may not exist in vLLM (v1 behavior)
If these are set in CLI/config for vLLM, they MUST be surfaced as ignored/deferred (choose one, but be consistent):
- `typical_p` (not present in vLLM chat request schema)
- `max_time` (not a vLLM chat request field)
- HF-only knobs like `num_beams` / `no_repeat_ngram_size` (unless we intentionally map them to vLLM equivalents; otherwise ignore)

## Server defaults that affect “knob behavior” (MUST document + expose)
vLLM will, by default, apply `generation_config.json` from the HF model repo if present, which can override defaults for sampling parameters.

To keep behavior predictable in this repo:
- add a vLLM server launch knob to control this, e.g. `backend.vllm.generation_config = "hf" | "vllm"`
  - `"hf"`: default vLLM behavior (applies `generation_config.json` if present)
  - `"vllm"`: pass `--generation-config vllm` at launch to disable HF generation_config defaults

## How profiles solve “server combos” (explicit)
With vLLM as an engine backend, profiles become the safe place for variations:
- same model, different vLLM launch knobs:
  - `profiles/maxctx.toml` sets `backend.vllm.max_model_len`
  - `profiles/tp2.toml` sets `backend.vllm.tensor_parallel_size = 2`
  - `profiles/gpu90.toml` sets `backend.vllm.gpu_memory_utilization = 0.90`

This keeps navigation consistent:
- you always look under `models/<model>/vllm/…` when you’re “running vLLM”.

## v2: persistable server mode (design hook)
Do not implement now, but keep the design compatible:
- `backend.vllm.mode` can later add:
  - `managed_persist`: do not stop server on exit
  - `external`: never start/stop server, just connect
- If we add persistence:
  - write a small state file under `~/.cache/model-runner/vllm/<model>/server.json`
  - store pid/port/base_url/served_model_name/model_path/start_time
  - validate “is alive + matches expected model” before reuse

## Acceptance criteria (v1)
- `tui Qwen3.5-9B vllm` starts vLLM, connects, and streams chat.
- Exiting the TUI stops the vLLM process reliably.
- Profiles work without moving files:
  - `tui Qwen3.5-9B vllm --profile maxctx` loads profile overrides.
- Repo navigation is consistent:
  - all local engines appear under `models/<model>/<engine>/...`.
