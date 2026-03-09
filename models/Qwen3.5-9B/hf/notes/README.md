# Qwen3.5-9B (HF, vision-capable)

## Where
- HF ID: `Qwen/Qwen3.5-9B`
- Local path: `/home/poop/ml/models/Qwen3.5-9B`
- HF cache revision (if known): (local folder)

## TL;DR
- What it’s good at: general chat + vision “image + text → text” prompts; supports tool-call formatting via its chat template.
- Biggest gotcha(s): it’s a multimodal checkpoint (`ForConditionalGeneration`), so you need a processor + multimodal auto-model (not `AutoModelForCausalLM`) if you want images.
- Recommended runner (`chat.py` vs `runner.py`): `tui.py` (HF backend) for `/image` + streaming thinking.

## What it is (from local `config.json`)
- Backend: HF / Transformers
- `model_type` / architecture: `qwen3_5` / `Qwen3_5ForConditionalGeneration`
- Vision token markers present: `vision_start_token_id=248053`, `vision_end_token_id=248054`
- Context/window: `text_config.max_position_embeddings=262144` (very large; practical limits depend on VRAM/kv-cache)

## Tokenizer / prompting (from local tokenizer files)
- `bos_token` / `eos_token` / `pad_token`:
  - `eos_token`: `<|im_end|>` (id `248046`)
  - `pad_token`: `<|endoftext|>` (id `248044`)
  - `bos_token`: none
- Chat template format (high level):
  - Roles are wrapped in Qwen-style `"<|im_start|>{role}\\n...<|im_end|>"`.
  - Images are injected as placeholder tokens in user content: `<|vision_start|><|image_pad|><|vision_end|>`.
  - Tools are documented in a system header, and tool calls are expected as an XML-ish block.
- Special tags present:
  - Thinking: `<think>` (single token id `248068`), `</think>` (single token id `248069`)
  - Tools: `<tool_call>...</tool_call>`, `<tool_response>...</tool_response>`

### System prompt behavior
- Does it inject a default system prompt if you don’t provide one?
- Language/identity defaults: none enforced by the tokenizer; behavior depends on checkpoint.
- Recommended system prompt for your use cases: keep it short; vision prompts tend to be longer already.

### Tool calling format (if applicable)
- Tool schema expected (OpenAI-like or custom): custom (template embeds `<tools>` JSON list in system; calls are *not* OpenAI JSON).
- Tool call emission format (text tags vs structured fields): text tags:
  - `<tool_call><function=name>...<parameter=...>...</parameter>...</function></tool_call>`
- Tool response injection format: tool messages are wrapped as `<tool_response>...</tool_response>` inside a synthetic `user` turn.
- Parsing considerations: treat this as a model-specific protocol; you’ll need an adapter to map it to OpenAI-style tool calls.

### Reasoning / “thinking” behavior (if applicable)
- Does it emit `<think>...</think>`?
- Yes (the chat template inserts a `<think>` prefix for generation by default).
- Are think tags special tokens (does `skip_special_tokens=True` remove them)? No; they’re normal tokens (but are single-token markers).
- Do you want to hide/store reasoning in logs/UI? Use TUI thinking panel (grey) and `/show last` for raw/think/answer splits.

Note: the shipped chat template has an `enable_thinking` switch. If it’s set false, it emits an *empty* `<think>\\n\\n</think>` prefix.
This repo doesn’t pass template kwargs yet, so you can’t toggle that without editing the template or adding a runner knob.

## Recommended settings
- Model author recommended defaults (if any):
- Practical defaults for your machine:
  - Primary machine reference: `docs/environments/primary-dev-machine.md` (RTX 4090 24GB VRAM, i9, 64GB RAM).
  - Start: `--dtype auto` (CUDA → fp16) and a conservative `--max-context-tokens` until you confirm KV-cache headroom.
  - If you OOM, try `-4bit` (CUDA required) or reduce context.

### “Max context”: what’s the real max?
This checkpoint advertises a very large window:
- Theoretical (model limit): `262,144` tokens (`text_config.max_position_embeddings` and tokenizer `model_max_length`).

Practical max is limited by **VRAM** because KV cache scales linearly with tokens.

For Qwen3.5-9B, from `text_config`:
- layers = `32`
- kv heads = `4`
- head dim = `256`

Approx KV-cache per token (fp16/bf16 KV) is:
`32 layers * 4 kv_heads * 256 head_dim * 2 (K+V) * 2 bytes ≈ 131,072 bytes/token` (≈ **128 KiB/token**)

Rule-of-thumb KV-cache sizes:
- `16k` tokens ≈ `2 GB` KV
- `32k` tokens ≈ `4 GB` KV
- `64k` tokens ≈ `8 GB` KV
- `128k` tokens ≈ `16 GB` KV
- `262k` tokens ≈ `32 GB` KV (**won’t fit** on a 24 GB card even if weights were free)

What this means on a 24 GB GPU:
- If you run **fp16/bf16 weights** (no quant), you likely top out around **~32k–64k** context depending on overhead (and lower when using vision inputs).
- If you run **4-bit weights** (`-4bit`), you free up more VRAM for KV; higher contexts (e.g. **64k+**) become plausible, but speed will degrade sharply at long contexts.

If you want to “max it out” safely, treat it as an experiment:
1) use `-4bit`
2) set `--max-context-tokens` to a big number (e.g. `65536`)
3) increase in steps (or binary search) until you hit OOM / slowdowns you don’t like

Also remember: `max_new_tokens` spends from the same total window: `prompt_tokens + max_new_tokens <= usable_context`.

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
# Text-only chat in the unified TUI
python tui.py --config Qwen3.5-9B

# Vision chat: attach an image, then send a prompt
#   /image /path/to/pic.png
#   What is in this image?

# Bypass /image and run HF directly (for quick sanity checks)
python - <<'PY'
from transformers import AutoProcessor, AutoModelForMultimodalLM
from PIL import Image
proc = AutoProcessor.from_pretrained('/home/poop/ml/models/Qwen3.5-9B')
model = AutoModelForMultimodalLM.from_pretrained('/home/poop/ml/models/Qwen3.5-9B', device_map='auto', torch_dtype='auto').eval()
messages = [{'role':'user','content':[{'type':'image'},{'type':'text','text':'Describe this image.'}]}]
prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = proc(text=prompt, images=[Image.new('RGB',(16,16),(255,0,0))], return_tensors='pt').to(model.device)
out = model.generate(**inputs, max_new_tokens=64)
print(proc.tokenizer.decode(out[0], skip_special_tokens=True))
PY
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
- `tui.py` currently supports image attachments via `/image` for HF only; other backends ignore/deny images.
- Template kwargs like `enable_thinking` aren’t wired through the runner yet.

## Model-specific proposed flags (if you choose to update the runners later)
- Proposed flag: `--hf-enable-thinking {true|false}` (template kwarg passthrough)
  - Why: Qwen’s chat template already supports it; could reduce “hung” feeling by emitting a closed empty think block.
  - Expected impact: less/no reasoning text (depends on checkpoint behavior).
