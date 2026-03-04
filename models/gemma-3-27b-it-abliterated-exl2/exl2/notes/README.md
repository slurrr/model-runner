# gemma-3-27b-it-abliterated-exl2 (EXL2) notes

## Where
- Local weights: `~/ml/models/gemma-3-27b-it-abliterated-exl2/`
  - Best: pass the full directory path in `model_id`.
  - Also supported: bare `model_id` + `model_path` (or `exl2_repo_path`) base directory in config.
- Runner: unified TUI `tui.py` / `tui` with `--backend exl2`
- Repo config: `models/gemma-3-27b-it-abliterated-exl2/exl2/config/config.json`

## What it is
- Architecture: Gemma 3 (text+vision config present; running text-style chat)
- Quantization: EXL2 (from model `config.json` → `quantization_config`)
  - `bits`: 4.0
  - `head_bits`: 8

## EXL2 mental model (how this backend works)
EXL2 (ExLlamaV2) runs a compiled CUDA extension (`exllamav2_ext`) and is very sensitive to:
- CUDA toolkit / PyTorch CUDA version alignment
- extension build cache state under `~/.cache/torch_extensions/`
- “engine” knobs that change memory behavior (cache type, max_seq_len, autosplit, flash-attn, graphs)

In the unified TUI:
- We build the prompt client-side (Jinja template) from the current chat `messages[]`.
- We then tokenize and stream the model output, routing `<think>...</think>` (or `assume_think`) through the UI’s think/answer panels.

## Templates
- Active template (current config): `/home/poop/ml/exllamav2/chat_template_gemma_exl2.jinja`
  - Gemma-style turn-token formatting (matches upstream ExLlamaV2 examples).
  - Ignores `role=system` messages by design (so `system`/`system_file` in this repo will not affect the prompt unless you switch templates).

Recommended template for this repo (so system prompts work and the template is versioned here):
- `models/gemma-3-27b-it-abliterated-exl2/exl2/templates/gemma_chat_sanitized.jinja`

Other template options:
- Model-provided template: `~/ml/models/gemma-3-27b-it-abliterated-exl2/tokenizer_config.json` → `chat_template`
  - This one *does* support injecting a system message as a prefix to the first user turn (recommended if you want a system prompt).
  - Caveat: the shipped template uses HF-style helpers like `raise_exception(...)` which may not be available in this repo’s minimal EXL2 Jinja renderer.
- Model-folder templates: `~/ml/models/gemma-3-27b-it-abliterated-exl2/chat_template.jinja` (and `chat_template.json`)
  - Often mirrors upstream ExLlamaV2 “Gemma mode” formatting.
  - Not automatically picked up by `--chat-template default` in this repo; if you want to try it, point `chat_template` directly at the file.
- Repo-local experiments: `models/gemma-3-27b-it-abliterated-exl2/exl2/templates/`
  - `gemma_chat_sanitized.jinja` is the recommended repo-local Gemma template for this model.
  - `current.jinja` is a generic fallback and is usually **not** correct for Gemma-family instruct models.

Important: prompt format matters a lot for Gemma-style instruction models.
If output is weird (early `<eos>`, repetitive, “dies” quickly, etc.), try switching to the model’s own template:
- Set `chat_template` to empty in config (or pass `--chat-template default`) so EXL2 tries `tokenizer_config.json`’s `chat_template`.

### How Gemma actually wants to be prompted (key detail)
This model ships a Gemma-family chat template in its weights folder:
- `~/ml/models/gemma-3-27b-it-abliterated-exl2/tokenizer_config.json` → `chat_template`

It uses special turn markers (Gemma-style), not “System:/User:/Assistant:”:
- `<start_of_turn>user\n...<end_of_turn>\n`
- `<start_of_turn>model\n` (assistant role is `model`)

Also: “system” content is handled as a **prefix to the first user turn** (not a distinct system role in the prompt).

If you use a plain transcript template for Gemma, it often looks like:
- “starts responding then dies” (early stop / bad turn boundary)
- or it “hangs” after some text (it may be emitting special tokens that decode to empty)

## Current config caveats (likely issues)
Check `models/gemma-3-27b-it-abliterated-exl2/exl2/config/config.json`:

- `max_seq_len=10240` is relatively high for a 27B model in EXL2.
  - If your prompt is non-trivial, you’ll hit the sequence ceiling and/or allocate a large cache.
  - While debugging stability, start with `max_seq_len=4096–8192`.
- `max_new_tokens=2048` is also aggressive while you’re still chasing stability issues.
  - For debugging, `256–1024` is a better first pass.
- `min_p` is `null` right now (good default).
  - If you do set `min_p`, it’s a probability-like filter and should usually be in the `0.0–0.2` range (common: `0.02–0.10`).
- `min_free_tokens=256` is a reasonable starting point.
  - If you set this very high, context trimming becomes more aggressive (to “reserve” room for the reply).
- `exl2_repo_path` should point to your ExLlamaV2 repo (e.g. `~/ml/exllamav2`) only if you haven’t installed it into the venv.
  - If ExLlamaV2 is already installed, this setting usually doesn’t matter.
  - If you set it, it should be the ExLlamaV2 *repo* path, not your weights folder.

## Symptom: “starts fast, then dies” (common EXL2 causes)
This symptom usually falls into one of these buckets:

1) **VRAM OOM or allocator failure mid-generation**
   - Even if the model “loads”, some caches/allocations may be lazy and grow as tokens are generated.
   - Fix/workarounds:
     - Reduce `max_seq_len` (big VRAM impact).
     - Reduce `max_new_tokens`.
     - Switch cache type: `cache_type=8bit` (large memory savings) or smaller `max_seq_len`.
     - Enable `low_mem=true` (may trade speed for memory).

2) **CUDA kernel crash / extension mismatch**
   - Often caused by a PyTorch CUDA version vs toolkit mismatch, or stale extension builds.
   - Workarounds:
     - Install Ninja in this venv: `pip install ninja`
     - Clear stale extension cache and rebuild: see `docs/exl2_setup.md`
     - Temporarily disable advanced paths:
       - `--no-flash-attn --no-graphs --no-xformers --no-sdpa`

3) **Prompt/template mismatch causing early EOS**
   - Model emits an EOS token early because it doesn’t recognize the chat format.
   - Workarounds:
     - Use model-provided chat template (`chat_template=""` / `default`).
     - Keep system prompt short while debugging (large system prompts can also destabilize).

4) **Sampler misconfiguration**
   - Bad values (notably `min_p`) can cause “no valid tokens” behavior.
   - Fix:
     - Set `min_p=null` (or `0.05`) and re-test.

## Debug checklist (fastest path)
1) Reduce to a stable baseline:
   - `max_new_tokens=256`
   - `max_seq_len=4096` (or smaller)
   - `min_free_tokens=256`
   - `min_p=null`
2) Disable advanced engine paths:
   - run with `--no-flash-attn --no-graphs --no-xformers --no-sdpa`
3) Switch to model template:
   - `--chat-template default` (or clear `chat_template` in config)
4) If it still “dies”, assume extension/cuda issues:
   - follow `docs/exl2_setup.md`
   - clear `~/.cache/torch_extensions/*exllamav2*`

## Think tags (what’s possible with this model + EXL2)

### How EXL2 “thinking” works in this repo
EXL2/ExLlamaV2 does not provide a separate structured “thinking channel”.
All output arrives as plain text chunks.

This repo’s TUI splits “thinking vs answer” purely by **parsing text markers** in the stream (via `ThinkRouter`):
- start markers: `<think>`, `<|begin_of_thought|>`, etc.
- end markers: `</think>`, `<|end_of_thought|>`, etc.
- `assume_think=true` starts the router in “thinking mode” and switches to “answer mode” only after an end marker appears.

### What this model does by default
This Gemma model does **not** ship a chat template that uses `<think>` tags:
- `~/ml/models/gemma-3-27b-it-abliterated-exl2/tokenizer_config.json` → `chat_template` contains `<start_of_turn>` / `<end_of_turn>`, but no
  `<think>` / `</think>`.

It *can still generate* `<think>` / `</think>` as plain text (they tokenize as normal text, not special “reasoning tokens”), but it will not
do so reliably unless you prompt it to.

Tokenization reality check (from the model’s tokenizer assets):
- `<start_of_turn>` and `<end_of_turn>` are single special tokens (Gemma control tokens).
- `<think>` / `</think>` are *not* special tokens here (they tokenize into multiple normal tokens).

Practical implications:
- You can still enable `show_thinking=true` (UI behavior), but the thinking panel may stay empty if the model never emits markers.
- `assume_think=true` is not useful unless the model reliably emits an end marker such as `</think>`.

### Getting “thinking panel” parity (recommended approach for this model)
If your goal is “make this model behave like the other think-tag models in the TUI”, the most reliable approach is:

1) Use the repo-local Gemma template that sanitizes history
- Keep: `models/gemma-3-27b-it-abliterated-exl2/exl2/templates/gemma_chat_sanitized.jinja`
  - This already strips any assistant history *before* the final `</think>` so you don’t keep replaying long reasoning into the next turn.

2) If the model emits only `</think>` (no `<think>`), enable “assume think”
- Switch template: `models/gemma-3-27b-it-abliterated-exl2/exl2/templates/gemma_chat_sanitized_assume_think.jinja`
  - This puts a literal `<think>` *in the prompt* right after `<start_of_turn>model\n` to nudge the model into a “thinking-first” style.
- Set `assume_think=true` so the UI routes early output into the thinking panel *even if the model never prints `<think>`*.

3) Add a small, explicit instruction (system prompt)
EXL2 cannot “hide” internal reasoning (there is no separate channel), so to see thinking you must ask the model to output it as text. Use a
simple convention that your router understands:
- “Write reasoning between `<think>` and `</think>`. After `</think>`, write the final answer.”

Concrete config/flags to try (minimal):
- `--show-thinking --assume-think`
- `--chat-template models/gemma-3-27b-it-abliterated-exl2/exl2/templates/gemma_chat_sanitized_assume_think.jinja`
- plus a system prompt line like the one above (either `--system '...'` or `system_file` in config)

Important constraints (EXL2/Gemma specifics):
- The backend stops on Gemma turn tokens (`<end_of_turn>`, `<start_of_turn>`) via **token IDs**. That’s correct for Gemma, but it means:
  - your model must emit `</think>` *before* it emits the `<end_of_turn>` token, or the UI will never switch from “thinking” to “answer”.
  - if the model sometimes forgets `</think>`, you’ll see “answer is empty” and all text in the thinking panel. That’s expected behavior.

This gives the router something to latch onto. It’s not private reasoning (it’s literal generated text), but it makes the UI behave like other
“thinking-tag” models.

### What you should *not* expect (and why)
This is a Gemma-family instruct model + EXL2 backend:
- There is no “OpenAI-style hidden reasoning channel”.
- There is no Ollama-style `message.thinking` field.
- Any “thinking” you see in the panel is plain text the model chose to generate, purely because the prompt asked for it.

## Why EXL2 can look like it “hangs” after a few tokens (Gemma turn tokens)
Gemma-style prompts rely on special tokens like `<end_of_turn>`.
Depending on decode settings, these can decode to **empty strings**, which means:
- the model is still generating tokens, but the UI prints nothing
- it feels like generation “died” or “hung”

Upstream ExLlamaV2’s `examples/chat.py` avoids this by using stop conditions that include **token IDs** for `<end_of_turn>` and friends.

In this repo, if this symptom persists even with a correct Gemma template, the likely missing feature is:
- EXL2 backend support for “stop token IDs” (not just stop strings), derived from the model’s tokenizer:
  - `tokenizer.eos_token_id`
  - `tokenizer.single_id("<end_of_turn>")`
  - `tokenizer.single_id("<start_of_turn>")`

If you want, we can add a tiny model-specific option for EXL2 like `exl2_stop_tokens: ["<end_of_turn>", "<start_of_turn>"]` which the backend
resolves to token IDs at runtime and never renders in the transcript.

## Known gaps vs upstream ExLlamaV2 chat (likely contributors to “buggy” feel)
Upstream `examples/chat.py` does a few things our unified TUI EXL2 backend does not (yet):

1) **Stop token IDs**
   - Upstream uses token IDs for `<end_of_turn>` / `<start_of_turn>` in Gemma mode.
   - Without this, the model can continue “past” the end of the assistant turn, sometimes producing tokens that decode to empty strings.

2) **`generator.full()` recovery**
   - Upstream detects when the sequence hits `max_seq_len` and restarts streaming with a rebuilt/truncated context.
   - Without this, generation can stop abruptly when the sequence fills.

3) **Token-native assistant history**
   - Upstream caches `responses_ids` (assistant token IDs) instead of retokenizing assistant text each turn.
   - This is more robust for special token boundaries and exact prompt formats.

If the model still “dies” even with a Gemma-correct template and sane sampler values, these are the top candidates to implement next.

## Useful commands
- Run EXL2 explicitly:
  - `tui --backend exl2 ~/ml/models/gemma-3-27b-it-abliterated-exl2`
- Capture a transcript (helps distinguish early EOS vs crash vs sampler issues):
  - set `save_transcript` in config to a path (e.g. `logs/exl2_gemma.jsonl`) and re-run
- If you suspect CUDA kernel errors:
  - `CUDA_LAUNCH_BLOCKING=1 tui --backend exl2 ...` (slower but surfaces errors earlier)
