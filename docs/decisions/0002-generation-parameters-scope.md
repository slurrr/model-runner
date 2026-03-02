# 0002 - Treat decoding knobs as `transformers`-level, not model-level

Date: 2026-02-28

## Context
When documenting “all parameters you can tweak”, it’s easy to conflate:
- checkpoint-specific settings (tokenizer/chat template, special tokens, tool-call format)
- generic decoding/sampling controls (top-k, top-p, beam search, stop strings, etc.)

Different checkpoints may ship different defaults (e.g. `generation_config.json`), but most decoding knobs are supported by `transformers` broadly.

## Decision
In HF model notes:
- Distinguish “checkpoint defaults” (what’s in the model folder) from “generation knobs” (what `transformers` supports).
- When needed, reference the repo’s current `.venv` `transformers.GenerationConfig` keys as the canonical list of decoding knobs.
- Keep a short “model-specific proposed flags” section that calls out which knobs are most valuable to expose in our runners for that particular checkpoint and use case (e.g. OpenClaw tool loops).

## Consequences
- We document top-k and other knobs even if they are not checkpoint defaults.
- We avoid implying a model “supports” a knob only if it appears in `generation_config.json`.
- We can decide which CLI flags to expose based on practical needs rather than guessing per model.

