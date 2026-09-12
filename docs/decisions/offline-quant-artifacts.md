# Offline Quant Artifacts

## Decision
Quantization artifacts are produced offline from this repo but stored outside the repo with the rest of the machine-local model weights.

## Why
- quantized artifacts are large and machine-local
- quantization is a prep workflow, not a runtime backend
- model-specific config still belongs in this repo next to notes and runtime config

## Rules
- keep compressor config under `models/<model>/vllm/compressor/`
- write artifacts under `~/ml/models/quants/<model>/<scheme>/...`
- preserve run history under `runs/`
- expose a stable `current` symlink for serving tools
- keep a manifest next to each saved artifact
