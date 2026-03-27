# Offline Compressor

Use `scripts/compress_model.py` to build vLLM-compatible FP8 artifacts outside the repo.

## Current status

The main repo environment is now anchored on `vllm>=0.18`.

Released `llmcompressor` builds do not currently resolve with `vllm 0.18.x` in the same environment because of upstream dependency conflicts around `compressed-tensors`, `torch`, and related packages.

Practical consequence:
- compression is currently **not** part of the main repo sync
- do not design the serving environment around compression
- treat compression as a deferred/offline workflow until a compatible released stack exists

OmniCoder-specific status and recipe rationale:
- [omnicoder-fp8-status.md](/home/poop/code/dev/model-runner/docs/reference/omnicoder-fp8-status.md)

## Quick start
1. Edit `models/<model>/vllm/compressor/default.toml`
2. Point `model.source` at the source weights
3. Run:

```bash
python scripts/compress_model.py --model OmniCoder-9B --scheme fp8
```

## Current contract
- Qwen3.5 and OmniCoder compression should not constrain the serving environment.
- The script loads the source model object directly so multimodal checkpoints stay `qwen3_5` instead of collapsing to `qwen3_5_text`.
- OmniCoder now defaults to `FP8_DYNAMIC` with `calibration.source = "none"`.
- `FP8_DYNAMIC` runs data-free in the current `llmcompressor` stack; the script must not force a calibration dataset or sequential calibration pipeline for that path.
- OmniCoder excludes the full visual tower from quantization via `re:^model\.visual($|\.)` and also excludes `lm_head`.
- Local calibration files can still be plain JSONL with a `text` column when using a scheme that actually requires calibration.

## Output layout
- `~/models/local/quants/<model>/fp8/runs/<timestamp-or-label>/<Compressed-Name>/`
- `~/models/local/quants/<model>/fp8/current`

`current` is a symlink to the latest successful artifact directory. Point vLLM or `agentmux` at that path if you want the most recent FP8 build.

## Useful overrides
```bash
python scripts/compress_model.py \
  --model OmniCoder-9B \
  --recipe-scheme FP8_DYNAMIC \
  --ignore 're:^model\\.visual($|\\.)' \
  --ignore lm_head \
  --run-label fp8-dynamic
```

For a calibration-requiring recipe, supply a real dataset and split explicitly.

## Dry run
Use `--dry-run` to print the resolved plan before compression:

```bash
python scripts/compress_model.py --model OmniCoder-9B --scheme fp8 --dry-run
```
