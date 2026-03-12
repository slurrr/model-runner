# <Model Name>

Optional (recommended for HF models): save the upstream Hugging Face model card as:
- `notes/model_card.md`

## Where
- HF ID:
- Local path:
- HF cache revision (if known):

## TL;DR
- What it’s good at:
- Biggest gotcha(s):
- Recommended runner (`chat.py` vs `runner.py`):

## What it is (from local `config.json`)
- Backend:
- `model_type` / architecture:
- Dtype in config:
- Context/window:

## Tokenizer / prompting (from local tokenizer files)
- `bos_token` / `eos_token` / `pad_token`:
- Chat template format (high level):
- Special tags present (e.g. `<think>`, tool-call tags):

### System prompt behavior
- Does it inject a default system prompt if you don’t provide one?
- Language/identity defaults:
- Recommended system prompt for your use cases:

### Tool calling format (if applicable)
- Tool schema expected (OpenAI-like or custom):
- Tool call emission format (text tags vs structured fields):
- Tool response injection format:
- Parsing considerations (multiple calls, JSON quirks, stop strings):

### Reasoning / “thinking” behavior (if applicable)
- Does it emit `<think>...</think>`?
- Are think tags special tokens (does `skip_special_tokens=True` remove them)?
- Do you want to hide/store reasoning in logs/UI?

## Recommended settings
- Model author recommended defaults (if any):
- Practical defaults for your machine:

### Tweakable runtime parameters (this repo’s current CLI flags)
`chat.py`:
- (list the flags you actually used/validated for this model)

`runner.py`:
- (list the flags you actually used/validated for this model)

### Generation parameters (cross-model reference)
- If you need “all possible generation knobs”, refer to `transformers.GenerationConfig` keys for the repo’s current `.venv`.
- List any generation parameters that are especially relevant for this model (e.g. `top_k`, `stop_strings`, `num_beams`).

## Suggested commands
```bash
python chat.py <name_or_path> ...
```

## Best use cases
- General chat:
- Coding/math:
- Tool/agent loops (OpenClaw):
- Other:

## Not ideal for
- Latency-sensitive:
- Clean assistant-only text:
- Other:

## OpenClaw integration notes (if relevant)
- Prompt construction notes:
- Tool call parsing:
- Tool output injection:
- Recommended stop criteria:

## Current runner limitations (repo-specific)
- Gaps you hit while using `chat.py` / `runner.py`:

## Model-specific proposed flags (if you choose to update the runners later)
- Proposed flag:
  - Why:
  - Expected impact:
