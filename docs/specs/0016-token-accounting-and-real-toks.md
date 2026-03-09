#!/usr/bin/env markdown
# Spec: Token accounting + real tok/s in the TUI (Phase 1)

Date: 2026-03-08

Status: Draft

## Context / problem
We need reliable token metrics in the TUI for:
- understanding context constraints (prompt + completion)
- diagnosing “thinking ate the budget”
- measuring throughput (tok/s) accurately

Today, token reporting is inconsistent:
- HF has a think-token counter, but not standardized prompt/completion/total.
- GGUF can tokenize but only uses it for think increments.
- EXL2 emits prompt token count, but not completion/total in a standard way.
- OpenAI/vLLM transport does not parse usage; think tokens are approximated from whitespace.
- `/show last` displays character lengths for raw/think/answer, which is not sufficient for token-based tuning.

Umbrella contract:
- `docs/specs/0015-backend-standardization-contracts.md`

ADR:
- `docs/decisions/0015-token-accounting-and-metrics.md`

## Goals
- Provide real token counts when the backend can provide them.
- Provide real tok/s (based on completion tokens and elapsed time).
- Keep overhead low; avoid retokenizing unless necessary.
- Keep behavior consistent across backends:
  - prefer backend-native truth
  - clearly show “unavailable” when not possible

## Non-goals (Phase 1)
- Perfect token accounting for arbitrary external OpenAI-compatible servers.
- A single universal tokenizer strategy across all backends.

## Data model changes (required)
Extend `TurnRecord` to include token counts when available:
- `token_counts: {prompt_tokens?: int, completion_tokens?: int, total_tokens?: int}`
- Optionally: `throughput: {tokens_per_s?: float}`

Also extend `/show last` output to include:
- `chars_*` and `tokens_*` (tokens optional)

Notes:
- Preserve backward compatibility: if token counts aren’t present, omit them or show `unavailable`.

## Backend implementation requirements

### 1. vLLM managed + OpenAI-compatible transport
Approach: parse backend-native usage from SSE frames.

Requirements:
- Add a vLLM managed launch option (config + template) to force usage in responses.
  - vLLM supports a “force include usage” flag (version dependent), and may support stream options.
- In `tui_app/transports/openai_http.py`, parse:
  - `frame["usage"]` when present
  - aggregate across frames (use final non-null usage as authoritative)
- Store in `TurnRecord.token_counts`.

If usage is unavailable:
- do not “fake” it with character counts.
- explicitly mark as `unavailable` in `/show`.

### 2. EXL2
Approach: use token IDs already exposed by the generator.

Requirements:
- Emit Meta keys:
  - `prompt_tokens` (already emitted today)
  - `completion_tokens` (count generated tokens)
  - `total_tokens = prompt_tokens + completion_tokens`
- Store these counts into `TurnRecord.token_counts`.

### 3. GGUF (llama.cpp)
Approach: use llama.cpp tokenization API (low overhead, already present).

Requirements:
- Prompt tokens:
  - for `create_chat_completion`, compute tokens by tokenizing the final rendered prompt (or use llama.cpp prompt token count if exposed).
- Completion tokens:
  - count emitted tokens incrementally via tokenization of appended output (preferred), or use engine counters if exposed by API.

### 4. HF
Approach: we control tokenizer and inputs; prompt token length is known.

Requirements:
- Prompt tokens: `input_ids.shape[-1]` after applying context limit/templating.
- Completion tokens: count generated tokens. In streaming mode, use streamer token IDs; in non-stream mode, use generated token IDs.

## UI / slash command requirements
- `/status`:
  - show last turn tokens when available
  - show tok/s when available
- `/show last`:
  - show `prompt_tokens`, `completion_tokens`, `total_tokens` when available
  - keep `len_raw/len_think/len_answer` but label them as `chars_*`

## Testing checklist
- vLLM managed:
  - usage present in server frames and is captured
  - tok/s matches rough expectation vs vLLM logs
- HF:
  - prompt token count changes as context trimming drops messages
- EXL2:
  - completion tokens stop at `max_new_tokens`
- GGUF:
  - token counts non-zero and stable across runs

