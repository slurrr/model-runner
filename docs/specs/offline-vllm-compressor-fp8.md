# Offline vLLM Compressor FP8

## Summary
- add a standalone offline compression script
- keep model-specific compressor config under `models/<model>/vllm/compressor/`
- write quantized artifacts outside the repo under `~/ml/models/quants`
- target vLLM-compatible FP8 artifacts first

## Contract
- compression is not a runtime backend and not a TUI feature
- this repo owns config, script, and notes for quantization
- `agentmux` or other runtimes consume the saved artifact path later
- each run writes to a versioned path and updates a stable `current` symlink

## Script
- entrypoint: `scripts/compress_model.py`
- input: model-local TOML with CLI overrides
- output: compressed artifact directory plus `compression-manifest.json`

## Output Layout
```text
~/ml/models/quants/<model>/fp8/runs/<timestamp>-<label>/<compressed-name>/
~/ml/models/quants/<model>/fp8/current -> runs/.../<compressed-name>
```

## Config Layout
```text
models/<model>/vllm/compressor/default.toml
```

Sections:
- `[model]`
- `[output]`
- `[calibration]`
- `[recipe]`

## Calibration
- local datasets and HF datasets are both valid inputs
- local inputs are expected to be formats supported by Hugging Face datasets loading
- FP8 defaults to `FP8_BLOCK` in the starter TOMLs
- recipe scheme, sample count, and sequence length are all configurable without changing the script

## Non-goals
- no AWQ or GPTQ in v1
- no launch-path integration
- no in-repo model artifact storage
