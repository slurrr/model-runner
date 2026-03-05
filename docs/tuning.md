# Tuning scratchpad (sampling + prompting)

This is a practical reference for dialing in model behavior **without** training (no LoRA/finetune).
Names and availability vary by backend (HF vs llama.cpp vs Ollama), but the underlying ideas mostly match.

## “OpenAI API compatible” but “may not emit JSON” (tool calls)

People use “OpenAI-compatible” to mean different things:

- **API surface compatible**: you can send `messages=[{role, content}, ...]` and stream tokens (chat-completions style).
- **Prompting compatible**: the model generally understands system/user/assistant roles and can follow “call a tool” instructions.
- **Tool-call schema compatible (strict)**: the model reliably emits a machine-parseable structure (typically JSON) matching a provided schema.

Many models are compatible with the *idea* of tool calls, but **not reliable at strict JSON emission**. That’s usually not “serialization”
inside the model; it’s just that next-token sampling doesn’t guarantee valid JSON unless you:

- constrain decoding (grammar / JSON mode / function-calling aware sampler), and/or
- make the output format extremely strict (and still handle occasional violations).

In practice: “can do tools” ≠ “always produces valid OpenAI `tool_calls` JSON”.

## Quick mental model

- **Prompt/template** sets *what the model tries to do* (persona, format, tool protocol).
- **Sampling parameters** set *how deterministic vs exploratory* it is.
- **Token limits** (`max_new_tokens` / `num_predict`) set *how far it can go* (and whether “thinking” consumes the budget).

## Core knobs (most useful)

### `temperature`
Controls randomness globally.

- Turn **down**: more deterministic, fewer tangents, more repetitive.
- Turn **up**: more creative, more variance, more risk of format breakage.
- Typical ranges:
  - “reliable assistant / tools”: `0.0–0.4`
  - “general chat”: `0.5–0.9`
  - “wild/creative”: `1.0+`

### `top_p` (nucleus sampling)
Samples from the smallest set of tokens whose cumulative probability is `top_p`.

- Turn **down**: tighter, safer, more consistent formatting.
- Turn **up**: broader vocabulary/ideas; can increase meandering.
- Typical ranges: `0.8–0.98` (tool-ish prompts often like `0.85–0.95`).

### `top_k`
Samples only from the top `k` tokens by probability (then applies other filters).

- Turn **down**: more conservative; can reduce nonsense.
- Turn **up**: more variety (but can increase drift).
- Typical ranges: `20–100` (some setups use `0` to disable top-k).

Note: some backends don’t expose `top_k` or treat it differently; absence of a knob doesn’t mean the *model* “lacks top-k” — it’s a
sampler setting, not a per-model capability.

### `min_p` (aka “minimum probability”)
Filters out tokens whose probability is below `min_p * p_max` (relative to best token).

- Turn **up**: prunes long-tail junk; can improve coherence and reduce “garbage tokens”.
- Turn **too high**: can make output terse/odd or get stuck.
- Typical ranges: `0.02–0.10` (many like `0.05`).

### Repetition controls
Names vary:
- HF: `repetition_penalty`, `no_repeat_ngram_size`
- llama.cpp: `repeat_penalty`, `repeat_last_n`
- Ollama: `repeat_penalty` (and sometimes `repeat_last_n`)

Effects:
- Increase penalty / increase window: less looping, less “Alex: Alex:” style repeats.
- Too much: can harm factuality and cause weird synonyms / stilted prose.

Typical starting points:
- `repeat_penalty`: `1.05–1.25`
- `repeat_last_n`: `128–1024` (bigger window = stronger anti-looping)

### Token budget: `max_new_tokens` / `num_predict`
Hard cap on generated tokens.

- If a model “thinks” a lot, the cap can be consumed by that, causing “cut off mid-thought” or no visible final answer.
- If you hide thinking via post-processing, it can *feel* like the model hung while it’s actually generating hidden tokens.

Rule of thumb: if you want short answers, prefer **prompting** (“be brief”) + a smaller token cap, but leave enough headroom for
multi-step tasks.

### Context window: `max_context` / `num_ctx`
How much prompt+history the backend keeps.

- Larger context helps long sessions, but increases latency and VRAM/RAM needs.
- For “agent” behavior, consider *history hygiene* (don’t feed back reasoning/thinking; summarize) before cranking context.

### `stop` / stop strings
Stop sequences end generation when encountered.

- Useful for preventing the model from continuing into role labels or extra sections.
- Backend semantics differ:
  - Ollama supports `stop: ["..."]` directly.
  - llama.cpp has stop strings in its API.
  - HF `generate()` needs custom stopping criteria for arbitrary strings (not universally available via a simple flag).

## GGUF / llama.cpp engine knobs (load-time)

These are primarily **performance / memory / context-scaling** controls. They don’t directly make a model “smarter”, but they can make it
usable (fit in VRAM), faster, and more stable for long chats.

### `n_ctx` (context window)
How many tokens llama.cpp allocates for the KV cache (prompt + history + current generation).

- Turn **up**: longer chat history fits, but KV cache VRAM cost grows (often the dominant VRAM consumer).
- Turn **down**: less VRAM pressure, but earlier chat turns get pushed out sooner.

### `n_batch` / `n_ubatch` (prompt eval batching)
These mostly affect **prompt ingestion speed** and VRAM usage during prompt eval.

- Higher `n_batch`: faster prompt processing (up to limits), but more VRAM usage and sometimes instability if too high.
- `n_ubatch` (micro-batch): can help smooth VRAM usage / scheduling; smaller can be more stable on tight VRAM budgets.

Rule of thumb starting points (varies by model + GPU):
- `n_batch`: `256–2048`
- `n_ubatch`: `64–512`

If you’re right at the VRAM cliff, try lowering `n_batch` first.

### `n_threads` / `n_threads_batch`
CPU threading settings for parts of llama.cpp that still touch CPU (tokenization, prompt eval orchestration, etc.).

- If you’re fully offloaded to GPU and the model is “GPU bound”, threads matter less.
- If you see CPU pegging or slow prompt eval, threads can matter a lot.

Rule of thumb:
- set `n_threads` to physical cores (or a bit below) and measure.

### RoPE / YARN context scaling knobs
These modify how the model handles positions beyond its training window. They can help with longer contexts, but can also hurt quality.

Commonly encountered knobs:
- `rope_scaling_type`
- `rope_freq_base` / `rope_freq_scale`
- `yarn_ext_factor`, `yarn_attn_factor`, `yarn_beta_fast`, `yarn_beta_slow`, `yarn_orig_ctx`

Practical guidance:
- Don’t change these unless you’re deliberately testing “longer-than-trained” context behavior.
- Change one variable at a time and keep a fixed eval prompt so you can compare.
- Expect model-specific behavior (some checkpoints degrade rapidly when pushed beyond training context).

## Advanced / backend-specific knobs (use selectively)

### `typical_p`
Alternative sampling filter that prefers “typical” tokens vs only high-probability tokens.

- Can reduce blandness without going full creative.
- Typical range: `0.8–0.99`.

### `tfs_z` (tail-free sampling, llama.cpp)
Prunes the “tail” differently from `top_p`.

- Can reduce weird rare-token excursions.
- Typical range: `0.90–0.99` (or `1.0` to disable).

### Mirostat (llama.cpp / some Ollama builds)
Adaptive sampler to target a “surprise”/perplexity level.

- Pros: can stabilize verbosity/creativity across prompts.
- Cons: can be harder to reason about; can fight strict formatting.
- Common starting points:
  - `mirostat`: `2`
  - `mirostat_tau`: ~`5–8`
  - `mirostat_eta`: ~`0.1`

### `seed`
Reproducibility for debugging.

- Use a fixed seed when you’re chasing a weird formatting bug or `</think>` leakage.

## Prompt/template levers (often bigger than sampling)

### System prompt “contract”
Put stable instructions in system (brevity, no emojis, format rules, tool protocol). Avoid repeating a “prefix” in every user message unless
you truly need per-turn injection.

### Few-shot examples
If you want tool calls or strict formatting, 1–2 examples often do more than parameter tweaks.

### History hygiene (important for “think tags”)
If a model outputs reasoning markers (`<think>...</think>` or just `</think>`), don’t feed that text back into history. Keep only the final
answer portion of assistant messages.

## Starter profiles (rough baselines)

These are not “best”; they’re quick starting points to iterate from.

### 1) Succinct, reliable assistant (good for agent/tool scaffolding)
- `temperature`: `0.2`
- `top_p`: `0.9`
- `top_k`: `40`
- `min_p`: `0.05`
- `repeat_penalty`: `1.15`
- `max_new_tokens` / `num_predict`: moderate (enough for the task), not huge
- Prompting: strong system contract, strict output format, minimal chat template

### 2) Rambling funny chatbot
- `temperature`: `0.9`
- `top_p`: `0.97`
- `top_k`: `100`
- `min_p`: `0.02`
- `repeat_penalty`: `1.05`
- Prompting: looser system prompt, allow style, allow jokes/emoji (if desired)

### 3) “Wise teacher” (explain + structure)
- `temperature`: `0.5`
- `top_p`: `0.92`
- `top_k`: `50`
- `min_p`: `0.05`
- `repeat_penalty`: `1.10`
- Prompting: system: “teach step-by-step, use headings + short examples, check understanding”

### 4) “Rebellious student” (opinionated, slightly contrarian)
- `temperature`: `0.7`
- `top_p`: `0.95`
- `top_k`: `80`
- `min_p`: `0.03`
- `repeat_penalty`: `1.08`
- Prompting: system: “push back politely, ask ‘why’ often, propose alternatives”

## Debug workflow (when behavior is inconsistent)

1. Fix the template first (role markers, missing delimiters, accidental literal speaker prefixes).
2. Fix the system contract (remove conflicting instructions; stop repeating per-user prefixes).
3. Lock `seed`, lower `temperature`, tighten `top_p` to stabilize reproduction.
4. Add stop strings to prevent role-label spillover.
5. Only then tweak advanced samplers (min_p / typical_p / mirostat).
