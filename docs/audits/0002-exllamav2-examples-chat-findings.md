# Findings: ExLlamaV2 `examples/chat.py` (chat interface patterns)

Date: 2026-03-03

This doc summarizes what ExLlamaV2’s upstream `examples/chat.py` does and what seems most relevant for this repo’s EXL2 backend behavior and troubleshooting.

Source(s) reviewed (local clone):
- `/home/poop/ml/exllamav2/examples/chat.py`
- `/home/poop/ml/exllamav2/examples/chat_prompts.py`
- `/home/poop/ml/exllamav2/examples/minimal_chat.py`
- `/home/poop/ml/exllamav2/chat_template_gemma_exl2.jinja`
- `/home/poop/ml/exllamav2/exllamav2/generator/base.py` (`full()`)

## 1) Upstream EXL2 chat loop: core design

### A) Token-native conversation history (not “messages → template → retokenize” each turn)
Upstream `examples/chat.py` does **not** keep a list of `{role, content}` messages and re-render a prompt string every turn.

Instead it stores:
- `user_prompts: list[str]` (each user turn already “role-decorated”, e.g. `"User: <text>"` in some modes)
- `responses_ids: list[Tensor]` (the assistant’s generated token IDs for each turn)

Then each turn it rebuilds the active context as **token concatenation**:
- encode user prompt text → token IDs
- append the cached assistant `responses_ids[turn]` (already tokenized)

Implications:
- avoids repeated prompt-string rendering and retokenization costs
- preserves special tokens exactly as generated (role markers, end-of-turn tokens, etc.)
- makes “history hygiene” easier to implement at the token level (e.g. drop thinking tokens, stop tokens)

### B) Prompt format is an explicit “mode” (model-specific formatting)
Upstream requires choosing a prompt format mode (`--mode`), implemented in `examples/chat_prompts.py`.

Each `PromptFormat_*` defines:
- `first_prompt(sysprompt: bool)` and `subs_prompt()`
- `stop_conditions(tokenizer)` (token IDs and/or strings)
- `encoding_options()` → `(add_bos, add_eos, encode_special_tokens)`

This matters because many models (Gemma, Llama3, ChatML, Qwen/QwQ, etc.) are extremely sensitive to the exact role markers and turn delimiters.

### C) Stop conditions are prompt-format-defined (token IDs + strings)
Upstream sets stop conditions like this:
- `sc = prompt_format.stop_conditions(tokenizer)`
- `generator.set_stop_conditions(sc)`

Notably, stop conditions include **token IDs** for end-of-turn markers (not only strings).

Example: Gemma mode stops on:
- `tokenizer.eos_token_id`
- `tokenizer.single_id("<end_of_turn>")`
- `tokenizer.single_id("<start_of_turn>")`

### D) Mid-generation “context full” recovery
Upstream checks:
- `if generator.full():` then rebuild context and restart streaming:
  - `active_context = get_tokenized_context(model.config.max_seq_len - min_space_in_context)`
  - `generator.begin_stream(active_context, settings)`

`generator.full()` is defined as:
- `sequence_ids.shape[-1] >= model.config.max_seq_len`

Implications:
- If the model runs out of sequence length mid-response, upstream attempts to recover (by truncating old turns and continuing generation).
- Without this, “starts fast then dies” can happen if generation hits the hard `max_seq_len` wall.

### E) Guardrails for unstable checkpoints
Upstream includes a “repeated token streak” hard stop:
- if one token repeats `N` times consecutively (default 64), stop the response.

This is meant to catch pathological loops that otherwise look like hangs.

## 2) Special case: Gemma formatting is not “System/User/Assistant: …”

Upstream includes a dedicated Gemma prompt format (`PromptFormat_gemma`) using special markers:

- first prompt:
  - `<bos><start_of_turn>user\n`
  - (optionally `<system_prompt>` injected inside the user block)
  - `<user_prompt><end_of_turn>\n<start_of_turn>model\n`
- subsequent prompts insert a literal `<end_of_turn>\n` before each new user turn

There is also a dedicated Jinja template in the repo:
- `/home/poop/ml/exllamav2/chat_template_gemma_exl2.jinja`

Key detail: that template explicitly **ignores system messages**, matching `PromptFormat_gemma`’s intent.

Takeaway:
- For Gemma-family EXL2 models, a generic “System: … / User: … / Assistant: …” template can cause early EOS, weird turn-taking, or seemingly “dead” generation.

## 3) What we can learn / apply (conceptually) to this repo

### A) Prompt format selection needs to be model-aware
Upstream doesn’t try to solve chat formatting with one generic template. It picks a format.

In our repo, the most reliable approach per model is usually:
1) Prefer the model’s own `tokenizer_config.json` `chat_template` (when present and correct).
2) If missing, provide a per-model template under `models/<model>/exl2/templates/current.jinja` that matches the model’s training format.
3) For Gemma specifically, use a Gemma-format template (start/end-of-turn tokens), not a plain role-labeled transcript.

### B) “Reserve space for reply” must be coherent with `max_new_tokens`
Upstream uses:
- `min_space_in_context = response_chunk` (default 250)
- context is built to `max_seq_len - min_space_in_context`
- response length is separately capped (`max_response_tokens`)

If `max_new_tokens` is larger than your reserved free space, you are likely to hit `generator.full()` mid-stream.

If your implementation does not include upstream’s `generator.full()` recovery behavior, the safe operational posture is:
- keep `max_new_tokens <= min_free_tokens` (or reserve more free space)

### C) Stop conditions should include token IDs, not only strings
Upstream uses token IDs for `<end_of_turn>` and similar markers.

If you only use string stops, you can miss cases where:
- the model emits the token but decoding doesn’t yield the exact string you expected, or
- the token boundary doesn’t align with the string

### D) Token-native history is robust
Storing assistant history as token IDs (like upstream) avoids:
- retokenization drift
- accidental whitespace normalization
- template bugs that accidentally re-wrap old assistant messages incorrectly

This is particularly relevant for models that embed special turn markers in the assistant output.

## 4) Practical troubleshooting checklist (based on upstream patterns)

When EXL2 “starts fast then dies”, try to identify which bucket you’re in:

1) **Prompt format mismatch (early EOS / no continuation)**
   - Symptom: response ends abruptly, often after a short phrase, with no error.
   - Action: use a format that matches the model.
     - For Gemma: use a Gemma-style template (see ExLlama’s `chat_template_gemma_exl2.jinja`).
     - Prefer the model’s own `tokenizer_config.json` `chat_template` where possible.

2) **Context length wall (`max_seq_len`)**
   - Symptom: response streams for a bit then stops when sequence length is exhausted.
   - Action: reduce `max_new_tokens`, reduce prompt/history, or increase `max_seq_len` (VRAM permitting).
   - Upstream behavior: detect `generator.full()` and restart stream with truncated history.

3) **Sampler misconfiguration**
   - Symptom: degeneracy (loops), or strange stalls.
   - Action: revert to upstream-ish sampler defaults:
     - `top_k ~ 50`, `top_p ~ 0.8–0.95`, `temperature ~ 0.7–1.0`
     - `min_p` should be a small float (or disabled), not a large integer.

4) **Unstable checkpoint / pathological loop**
   - Symptom: repeated output / no progress.
   - Upstream uses a repeated-token streak cutoff to stop.

5) **Kernel crash / CUDA mismatch**
   - Symptom: hard stop or exception; sometimes only visible with `CUDA_LAUNCH_BLOCKING=1`.
   - Action: see `docs/exl2_setup.md`.

## 5) Model-specific note for Gemma EXL2

If you’re running a Gemma-family EXL2 model in this repo and using a generic role-labeled template:
- Consider switching to a Gemma-style template based on ExLlama’s `chat_template_gemma_exl2.jinja`.

That template intentionally:
- ignores system messages
- wraps each user turn in `<bos><start_of_turn>user ... <end_of_turn><start_of_turn>model`
- appends assistant content verbatim

This is closer to what the model expects than “System: … / User: … / Assistant: …”.

