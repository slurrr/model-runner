# OmniCoder FP8 Status

## Current Recommendation

This document describes the recipe shape, not the active Fedora serving environment.

The repo’s main environment is now anchored on `vllm>=0.18`, and released `llmcompressor` builds do not currently coexist with that stack in one environment.

The current recommended OmniCoder path is no longer the old calibrated `FP8` recipe.

Use:
- `scheme = "FP8_DYNAMIC"`
- `targets = "Linear"`
- `ignore = ["lm_head", "re:^model\\.visual($|\\.)"]`
- `calibration.source = "none"`

This recommendation is based on three verified facts from the current local environment:
- the local OmniCoder checkpoint is a `Qwen3_5ForConditionalGeneration` multimodal model with a full `model.visual` tower and `model.visual.merger`
- the installed `llmcompressor` version is `0.10.0.1`
- the current `llmcompressor` docs and entrypoint README describe `FP8_DYNAMIC` as the standard W8A8 FP8 path and explicitly state that no calibration dataset is required for RTN / dynamic activation quantization

## Why The Old OmniCoder Recipe Failed

The old recipe was:
- `scheme = "FP8"`
- `targets = "Linear"`
- `ignore = ["lm_head", "re:.*mlp.gate$"]`
- calibration data from plain JSONL built out of SWE-smith trajectories

That path had two fundamental problems:
- it quantized the visual tower because the ignore list never excluded `model.visual.*`
- it treated calibration shaping as the main recovery lever even though the current library’s recommended high-quality FP8 path is `FP8_DYNAMIC`, which does not require a calibration dataset in the first place

There was also a smaller but important correctness issue:
- OmniCoder’s dense MLP modules are named `mlp.gate_proj`, not `mlp.gate`, so the previous ignore rule did not exclude anything relevant in the dense text backbone

## Verified OmniCoder Module Shape

A local module-tree audit of the checkpoint shows:
- vision modules live under `model.visual.*`
- the vision merger also lives under `model.visual.merger.*`
- text modules live under `model.language_model.layers.*`
- text layers include both `linear_attn.*` and `self_attn.*` linear projections
- dense MLP modules are `mlp.gate_proj`, `mlp.up_proj`, and `mlp.down_proj`

That means the exclusion we actually need for the multimodal-safe FP8 recipe is the visual subtree, not a guessed gate rule.

## Current Script Contract

`scripts/compress_model.py` now supports all three calibration modes explicitly:
- `local`
- `hf`
- `none`

For `FP8_DYNAMIC` and `FP8_BLOCK`:
- the script allows `calibration.source = "none"`
- it does not force a dataset
- it does not force the sequential calibration pipeline

For calibration-requiring schemes such as plain `FP8`:
- the script still requires a real dataset and split
- local JSONL staging is still supported

## Recommended Command

```bash
python scripts/compress_model.py --model OmniCoder-9B --scheme fp8
```

The current default config already resolves to the recommended OmniCoder recipe.

## What Is Still Missing

One thing is still not proven yet in-repo:
- empirical quality validation of the resulting OmniCoder FP8_DYNAMIC artifact against the source checkpoint on a representative coding/chat prompt set

The recipe is now derived from the actual checkpoint structure and current upstream guidance, not from guesswork.
The remaining work is validation, not further recipe speculation.

## Serving-env note

For now, keep the repo’s main `.venv` optimized for vLLM serving and general runtime work.
If compression is needed before upstream packages converge, handle it as a separate one-off workflow rather than redefining the main project environment around it.
