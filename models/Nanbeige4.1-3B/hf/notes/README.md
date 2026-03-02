# Nanbeige4.1-3B

## Where
- HF ID: `Nanbeige/Nanbeige4.1-3B`
- Local path: `/home/poop/ml/models/Nanbeige4.1-3B`
- HF cache revision: `6f3b2c34ac928f8b27849d92a185b9a4af59be63` (from `~/.cache/huggingface/hub`)

## TL;DR
- This checkpoint is a “reasoning + agentic tools” oriented chat model: it may emit a `<think>...</think>` block before the final answer.
- The tokenizer chat template has **first-class tool calling** support and uses an **OpenAI-like tool schema** (`{"type":"function","function":{...}}`), but tool calls are emitted as XML-ish tags: `<tool_call>{...json...}</tool_call>`.
- If you don’t provide a system message, the template injects a default system prompt in Chinese (either “Nanbeige identity” or “tool calling expert”), which can bias behavior.

## What it is (from local `config.json`)
- Backend: Hugging Face (`transformers`) via `chat.py` / `runner.py`
- `model_type`: `llama` (`LlamaForCausalLM`)
- Dtype in config: `bfloat16`
- Context: `max_position_embeddings=262144` (very large)

## Tokenizer / prompting (from local tokenizer files)
- Chat template uses Qwen-style separators:
  - `bos_token=<|im_start|>` (`bos_token_id=166100`)
  - `eos_token=<|im_end|>` (`eos_token_id=166101`)
  - `pad_token=<unk>` (`pad_token_id=0`)
- The vocab includes reasoning/tool tags:
  - `<think>` / `</think>` are present (IDs `166103` / `166104`) but are **not** “special tokens”, so `skip_special_tokens=True` will **not** remove them.
  - `<tool_call>` / `</tool_call>` are present (IDs `166105` / `166106`) and are used by the chat template to wrap tool call JSON.

### What the `<think>` tags mean in practice
- Expect the model to sometimes produce:
  - a reasoning block (inside `<think>...</think>`) and then
  - the “final” assistant message afterward.
- The provided chat template includes logic to split assistant content on `</think>` when building the next prompt. That’s a hint that the intended flow is: reasoning may exist, but you don’t necessarily want to feed all prior reasoning back into later turns.
- If you’re capturing transcripts, store both when debugging; if you’re building a UX, you may want to hide the reasoning block (presentation-layer choice).

### System prompt behavior (important)
This model’s default chat template always emits a leading `system` message:
- If your first message is `{"role":"system", ...}`, it uses your content.
- If not:
  - **No-tools mode:** it injects a default identity system prompt in Chinese (roughly “You are Nanbeige…”).
  - **Tools mode:** it injects a default “tool calling expert” system prompt in Chinese and then prints tool signatures/instructions.

Implication for your runners/integrations:
- For English / OpenClaw / “general assistant” use, you should provide an explicit system message as the first turn (even a short one) to avoid inheriting the Chinese default.

### Tool calling format (and how close it is to OpenAI)
The template’s *tool definitions* are OpenAI-shaped:
- `tools` is expected to look like a list of objects shaped like:
  - `{"type":"function","function":{"name": "...", "description": "...", "parameters": {...}}}`

The template’s *tool call output* is **not** an OpenAI API `tool_calls` field; it is text that includes tags:
```text
<tool_call>
{"name": "SearchWeather", "arguments": {"location": "Beijing"}}
</tool_call>
```

Tool results are represented as `role="tool"` messages, and the template wraps them into:
```text
<tool_response>
...tool output...
</tool_response>
```

Compatibility takeaway for OpenClaw:
- Tool schema: mostly compatible with OpenAI-style definitions.
- Tool invocation: you likely need a small parser that extracts `<tool_call> ... </tool_call>` JSON blobs and executes them.
- Tool response injection: feed tool outputs back as `{"role":"tool","content": ...}` (the template handles wrapping).

### Alternate chat template shipped with this checkpoint
The model folder includes `tokenizer_config_search.json` with a much simpler template (no tools formatting, no tool-response wrappers). The model README mentions switching to it for “deep-search”.
Practical use: if you want a minimal chat format without the tool preamble, you can use that template (but you’ll lose the “tool calling” instruction formatting).

How to switch (requires code or a small helper script; `chat.py` does not currently have a flag for this):
```python
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/home/poop/ml/models/Nanbeige4.1-3B")
tok.chat_template = json.load(open("/home/poop/ml/models/Nanbeige4.1-3B/tokenizer_config_search.json"))["chat_template"]
```

Note: this may reduce prompt-side scaffolding, but it does not guarantee the model will stop emitting `<think>` blocks (those tokens are part of the checkpoint’s conventions).

## Recommended settings (from the model’s local `README.md`)
- Temperature: `0.6`
- Top-p: `0.95`
- Repeat penalty: `1.0`
- Max new tokens: very large (their docs mention `131072` for long reasoning)

### All tweakable runtime parameters (this repo’s current CLI flags)
These are the knobs you can change without editing code.

`chat.py`:
- `--config` (default empty; allows loading defaults from a JSON config)
- `--max-new-tokens` (default `2048`)
- `--stream` (stream token-by-token)
- `--temperature` (default `0.7`)
- `--top-p` (default `1.0`)
- `--top-k` (default unset)
- `--typical-p` (default unset)
- `--min-p` (default unset)
- `--repetition-penalty` (default `1.0`)
- `--max-time` (default unset)
- `--num-beams` (default `1`)
- `--no-repeat-ngram-size` (default `0`)
- `--stop-strings <s1> <s2> ...` (default unset)
- `--dtype {auto,float16,bfloat16,float32}` (default `auto`)
- `--system "<text>"` (default empty)
- `--system-file <path>` (used if `--system` is empty)
- `--user-prefix "<text>"` (prepended to each user turn)
- `--max-context-tokens <n>` (trim oldest turns to fit budget)
- `-4bit` / `-8bit` (CUDA only; mutually exclusive)
- `--hide-think` (strip `<think>...</think>`-style blocks from the assistant text)
- `--strict-think-strip` (more aggressive prefix stripping; use only if needed)

`runner.py`:
- `--config` (default empty)
- `--max-new-tokens` (default `200`)
- `--stream` (stream token-by-token)
- `--temperature` (default `0.7`)
- `--top-p` (default `1.0`)
- `--top-k` (default unset)
- `--typical-p` (default unset)
- `--min-p` (default unset)
- `--repetition-penalty` (default `1.0`)
- `--max-time` (default unset)
- `--num-beams` (default `1`)
- `--no-repeat-ngram-size` (default `0`)
- `--stop-strings <s1> <s2> ...` (default unset)
- `--prompt-prefix "<text>"` (prepended to each raw prompt before tokenization)
- `-4bit` / `-8bit` (CUDA only; mutually exclusive)
- `--hide-think`
- `--strict-think-strip`

### Model/tokenizer parameters that matter (present in local files, but not “CLI knobs”)
- Chat format is defined by the tokenizer’s `chat_template` (see `tokenizer_config.json` and `tokenizer_config_search.json` in the model folder).
- Special tokens in this checkpoint:
  - `bos_token=<|im_start|>` / `eos_token=<|im_end|>` / `pad_token=<unk>`
- Stop token: generation can stop early if `<|im_end|>` is produced (its token id is `166101` in this checkpoint).

### “All possible parameters” clarification (model vs generation)
Most “knobs” you’re thinking of (like **top-k**) are not properties of a specific checkpoint; they are generic `transformers` **generation** parameters (sampling/decoding controls) that work across most causal-LMs. A model *may* ship recommended defaults in `generation_config.json`, but this checkpoint’s `generation_config.json` only sets token IDs (BOS/EOS/PAD).

So, for “every single parameter that can be tweaked” you generally want:
- **Generation parameters** (generic, cross-model): `GenerationConfig` / `model.generate(...)` kwargs.
- **Prompting parameters** (model-specific): tokenizer `chat_template`, special tokens, tool-call format, etc.
- **Model config** (model-specific, usually not touched at runtime): values in `config.json` (e.g. RoPE settings, max positions).

### All generation parameters supported by this repo’s `transformers` (for reference)
In this repo’s `.venv`, `transformers` exposes these `GenerationConfig` keys (these can be set via `model.generate(..., **kwargs)` or `model.generation_config.<key> = ...` in code; your CLI may or may not expose them yet):

```text
assistant_confidence_threshold
assistant_early_exit
assistant_lookbehind
bad_words_ids
begin_suppress_tokens
bos_token_id
cache_config
cache_implementation
compile_config
constraints
decoder_start_token_id
disable_compile
diversity_penalty
do_sample
dola_layers
early_stopping
encoder_no_repeat_ngram_size
encoder_repetition_penalty
eos_token_id
epsilon_cutoff
eta_cutoff
exponential_decay_length_penalty
force_words_ids
forced_bos_token_id
forced_eos_token_id
guidance_scale
is_assistant
length_penalty
low_memory
max_length
max_matching_ngram_size
max_new_tokens
max_time
min_length
min_new_tokens
min_p
no_repeat_ngram_size
num_assistant_tokens
num_assistant_tokens_schedule
num_beam_groups
num_beams
num_return_sequences
output_attentions
output_hidden_states
output_logits
output_scores
pad_token_id
penalty_alpha
prefill_chunk_size
prompt_lookup_num_tokens
remove_invalid_values
renormalize_logits
repetition_penalty
return_dict_in_generate
sequence_bias
stop_strings
suppress_tokens
target_lookbehind
temperature
token_healing
top_h
top_k
top_p
typical_p
use_cache
watermarking_config
```

Notes:
- **top-k exists** (`top_k`) and is exposed as `--top-k` in this repo.
- Some of the more advanced keys (assistant/DoLa/watermarking/etc.) require extra setup and are typically configured in code rather than exposed as CLI flags.
- `generate()` also accepts non-config hook arguments (notably `logits_processor`, `stopping_criteria`, `prefix_allowed_tokens_fn`, `streamer`) which are powerful but not practical as CLI flags.

If you want to keep the notes future-proof, a good pattern is:
- “Checkpoint defaults”: what’s actually in the model folder (`generation_config.json`, tokenizer files, config).
- “Generation knobs worth exposing in this repo”: a short, curated list (see below).

### Knobs likely worth exposing for this model / OpenClaw work
Most core decoding knobs are already exposed as CLI flags in this repo. For OpenClaw/tool work the bigger “missing knobs” are tool-aware flow controls (see next section).

### Model-specific proposed flags (if you choose to update the runners later)
These are suggestions tailored to *this* checkpoint’s tool/thinking conventions (not “generic nice-to-haves”).

- `chat.py`: `--tools <path.json>` (or `--tools-json '<json>'`)
  - Why: this tokenizer template has a dedicated `tools` branch and expects OpenAI-like tool schemas; having a CLI path makes it easy to test tool formatting.
- `chat.py`: `--tool-mode {off,parse-only,execute}` (default `off`)
  - Why: this model emits `<tool_call>{json}</tool_call>` blocks; a mode flag makes it easy to switch between raw chat and agent loop behavior.
- `chat.py`: `--chat-template {default,search}` (or `--chat-template-file <path>`)
  - Why: this checkpoint ships an alternate template in `tokenizer_config_search.json`; exposing it as a flag makes experiments reproducible without editing code.
- `chat.py`: `--hide-think-behavior {strip,warn-on-empty,store-raw}`
  - Why: this checkpoint can generate long `<think>` blocks that hit `max_new_tokens`; “strip” can yield empty visible replies unless you handle the fallback explicitly.

In this repo, `runner.py`’s default `--max-new-tokens 200` is often too small for this model’s “reasoning-first” style, and even `chat.py`’s default `2048` may be insufficient for some prompts.

## Common symptom: “thoughts” then cut off mid-thought

### What it usually is
- The model is emitting a reasoning block (often wrapped with `<think>...</think>`, or sometimes plain reasoning-like text) before its final answer.

### Why it cuts off
- Most commonly: `--max-new-tokens` is too low, so generation ends before it reaches the final answer.
- Less commonly: the model may emit the EOS token (`<|im_end|>`) earlier than you expect, which stops generation immediately.

### Workarounds
- Increase `--max-new-tokens` substantially (start with `1024` or `2048`; go higher for complex tasks).
- If you don’t want to see the reasoning block, strip `<think>...</think>` in a post-processing step (whether that is in your script, OpenClaw integration, or UI).

## Suggested commands (local path auto-resolves from `~/ml/models/<name>`)

Chat (recommended for this model):

```bash
python chat.py Nanbeige4.1-3B --temperature 0.6 --top-p 0.95 --repetition-penalty 1.0 --max-new-tokens 2048
```

Text generation (not template-aware; expect more prompt sensitivity):

```bash
python runner.py Nanbeige4.1-3B --temperature 0.6 --top-p 0.95 --repetition-penalty 1.0 --max-new-tokens 2048
```

## What this model is best for (practical)
- Tool-using agent loops (OpenClaw-style): its chat template and tags are explicitly built around tool signatures, tool calls, and tool responses.
- Reasoning-heavy tasks where you don’t mind extra “scratchpad” tokens (math, coding, multi-step planning), especially if you can afford higher `--max-new-tokens`.
- “Deep-search” style multi-step workflows (the upstream docs explicitly position it there).

## What it’s *not* naturally optimized for
- Very short, latency-sensitive “one-liner” chat responses (it may spend tokens “thinking” before answering).
- Clean assistant-only text without any tool/thought conventions unless you:
  - provide a clear system prompt and/or
  - post-process out `<think>` blocks and tool tags.

## OpenClaw integration notes (concrete)
- Prompt construction:
  - Use the tokenizer’s chat template with a **system message first** (to override the default Chinese system).
  - Pass `tools=[...]` using the OpenAI-ish schema shown above so the template enters its tool-aware mode.
- Parsing tool calls:
  - Look for `<tool_call> ... </tool_call>` blocks and parse the JSON payload.
  - Be prepared for multiple tool calls in one assistant turn.
- Feeding tool results:
  - Add tool outputs as `role="tool"` messages; the template wraps them in `<tool_response>...</tool_response>`.
- Stopping criteria:
  - In a tool step, you may want to stop generation right after a `</tool_call>` appears (otherwise the model may “keep talking” after requesting a tool).

## Current runner limitations (repo-specific)
- `chat.py` is template-aware but is currently a plain chat loop (no tool definitions passed in, no tool-call parsing/execution). OpenClaw/tool support requires additional glue code.
- `runner.py` is a raw text generation loop (no chat template), so it won’t exercise the model’s tool-call conventions unless you manually include the formatted prompt yourself.

## Notes for this repo’s runners
- This model’s tokenizer reports an extremely large `model_max_length` value; long chats can grow very large, so if you hit OOM, shorten history or add truncation logic.
- If you see `<think>` tags in output, it’s expected for this checkpoint; stripping them is presentation, not a “model fix”.
