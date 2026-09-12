# Local Layout

`model-runner` stays model-first in the repo, but actual assets and generated state live outside the repo.

## Repo vs Machine Layout

### In repo

Use `models/<model>/<backend>/` for:
- `config/`
- `notes/`
- `prompts/`
- `templates/`
- `compressor/` recipes and backend-specific authored metadata

This is the control plane.

### Outside the repo

Use the home-directory layout for machine state:

- `~/models/local/`
  - local model folders you intentionally keep
- `~/models/local/gguf/`
  - GGUF artifacts
- `~/models/local/exl2/`
  - EXL2 artifacts
- `~/models/local/quants/`
  - quantized/compressed outputs
- `~/models/hf/`
  - Hugging Face managed cache
- `~/data/model-runner/`
  - calibration datasets and durable evaluation inputs
- `~/runs/model-runner/`
  - telemetry, logs, transcripts, outputs, state

This is the data plane.

## Runtime Path Contract

Runtime artifacts should be organized by logical identity, not by repo path.

Use:

- `~/runs/model-runner/logs/<backend>/<model>/<slot>/`
- `~/runs/model-runner/state/<backend>/<model>/<slot>/`
- `~/runs/model-runner/outputs/`
- `~/runs/model-runner/telemetry/`
- `~/runs/model-runner/transcripts/`

Examples:

- `~/runs/model-runner/logs/vllm/qwen3.5-9b/default/vllm-engine.stdout.log`
- `~/runs/model-runner/logs/vllm/qwen3.5-9b/default/vllm-engine.stderr.log`
- `~/runs/model-runner/state/vllm/qwen3.5-9b/default/vllm-managed.json`

Do not mirror repo-internal paths like `models/<model>/<backend>/config/...` under `~/runs`.
The repo layout is for authored control-plane files; the run layout is for machine-local runtime state.

## Why This Split Works

It preserves the useful part of the old repo design:
- model-first notes and config are still easy to find

while removing the bad part:
- runtime clutter and heavy artifacts no longer live in the repo

## Practical Rule

Ask:

- "Is this something I authored and want to version?"
  - keep it in the repo

- "Is this a weight file, quant, dataset, transcript, telemetry stream, or run artifact?"
  - keep it outside the repo
