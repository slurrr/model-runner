# Audit: `chat.py` `--hide-think` behavior (Nanbeige4.1-3B)

Date: 2026-02-28

## Summary of reported behavior

- Without `--hide-think`: Nanbeige4.1-3B emits very long “thinking” (`<think> ...`) and the generation often ends before a final answer appears (even with `--max-new-tokens 2048`).
- With `--hide-think`: you sometimes see:
  - first reply is non-empty,
  - subsequent replies are empty strings (assistant prints nothing), repeatedly.

## What `--hide-think` does today (code paths)

`chat.py` has two generation/display paths:

### 1) Non-streaming path (default)
- The model generates the full completion.
- New tokens are decoded into `raw_text`:
  - `raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()` (`chat.py:507`)
- If `--hide-think`, it post-processes:
  - `shown_text = strip_think_text(raw_text, ...).strip()` (`chat.py:509-510`)
- It prints `shown_text` (`chat.py:511`) and **stores `shown_text`** in conversation history:
  - `messages.append({"role": "assistant", "content": shown_text})` (`chat.py:513`)

### 2) Streaming path (`--stream`)
- Uses `TextIteratorStreamer(..., skip_special_tokens=True)` (`chat.py:462-463`)
- If `--hide-think`, it incrementally filters with `ThinkFilter.feed(piece)` and prints only non-think text (`chat.py:469-487`).
- At end it calls `ThinkFilter.flush()` (`chat.py:489-494`).
- It builds `shown_text = "".join(shown_parts).strip()` (`chat.py:498`) and stores it in history (`chat.py:513`).

## Why “empty assistant messages” happen with this implementation

The current `ThinkFilter` is designed to remove `<think>...</think>` blocks entirely. Two cases lead to an empty `shown_text` even when `raw_text` is non-empty:

### Case A: The model outputs only thinking (no final answer outside `<think>`)
Example output shape:
```text
<think>
...lots of reasoning...
</think>
```
If there is no content after `</think>`, stripping removes everything and returns an empty string. That’s “working as implemented”, but can be surprising.

### Case B (most likely with your symptom): the model is cut off mid-`<think>` (no closing tag)
Example output shape when `max_new_tokens` is exhausted:
```text
<think>
...lots of reasoning...
```
In this situation:
- `ThinkFilter` enters `in_think=True` when it sees `<think>` (`chat.py:82-83`).
- If the closing marker `</think>` never arrives, `flush()` returns `""` when `in_think` is still true (`chat.py:104-107`).
- Result: `shown_text` becomes empty even though the model produced many tokens (they were all inside an unclosed think block).

This matches the pattern: “without hide_think I see tons of thinking tokens and it cuts off; with hide_think I see empty messages.”

### Case C: perceived “empty output” while streaming due to prefix buffering
In streaming mode, `ThinkFilter` starts in an “implicit prefix” buffering mode (`chat.py:29-65`):
- it buffers output until it sees an end marker (like `</think>`), or
- the buffer exceeds ~8192 characters, at which point it releases the buffered text into the normal stripper.

For models that start a response with `<think>` and produce a very long reasoning segment, this means:
- you will see **no streamed output** for a long time (by design), and
- if the model never reaches `</think>` before `max_new_tokens` ends, you may see no output at all.

If you also use `--strict-think-strip`, the probe limit becomes “no limit” (`strict_prefix_strip=True` sets `implicit_prefix_probe_limit=None`), so streaming can stay silent indefinitely unless an end marker appears.

## Secondary issue specific to `chat.py`: storing the stripped text in history

`chat.py` stores `shown_text` (not `raw_text`) into `messages` (`chat.py:513`).

When stripping yields `""`, the conversation history gains assistant turns with empty `content`. That can:
- change the prompt the tokenizer chat template builds,
- potentially increase the chance of repeated “think-only then cut off” behavior,
- generally make debugging harder (because you lose the raw model output that explains what happened).

This persistence problem does **not** exist in `runner.py` because it does not maintain a message history (it only prints the decoded text).

## Interaction with Nanbeige4.1-3B’s tokenizer chat template

From the checkpoint’s `tokenizer_config.json`, the chat template explicitly understands:
- `<think>...</think>` blocks (it contains logic to split on `</think>` and handle “reasoning_content” vs “content”)
- tool calls as inline tagged JSON (`<tool_call>...</tool_call>`)
- tool responses (`<tool_response>...</tool_response>`)

Implication:
- Nanbeige’s own template expects these tags to appear in assistant messages, and has logic to manage them.
- Stripping tags before storing messages removes information the template could have used to structure the next prompt.
- However, the immediate “empty message” symptom still primarily points to “the model did not produce any non-think output within the token budget”.

## Is switching to `tokenizer_config_search.json` likely to “eliminate `<think>`”?

The alternate template in `tokenizer_config_search.json` is much simpler (no tool formatting branch, no reasoning extraction logic).

But:
- `<think>` is part of the model’s vocabulary and training conventions.
- Removing prompt-side reasoning/tool scaffolding **may** reduce how often the model emits `<think>`, but it is not guaranteed to eliminate it.

So switching templates is worth trying as an experiment, but it should not be treated as a guaranteed fix for “think-only outputs”.

## Suggestions (audit only; not implemented)

### 1) Make `--hide-think` degrade gracefully when it would print/store `""`
If `raw_text` is non-empty but `shown_text` becomes empty:
- print a one-line warning like:
  - `"[hide-think] Model output contained only <think>... (or was cut off). Increase --max-new-tokens or adjust prompt."`
- optionally store `raw_text` in history while printing the stripped output (see next suggestion).

### 2) Store `raw_text` in history, but print `shown_text`
Instead of storing `shown_text` into `messages`, store `raw_text` so:
- you keep full fidelity for debugging,
- Nanbeige’s template can apply its own “reasoning vs content” logic on subsequent turns,
- empty visible outputs don’t turn into empty-history turns.

If you still want to keep prompts smaller, you can store a transformed version:
- keep content after `</think>` if present,
- otherwise store a short sentinel or truncated reasoning.

### 3) Add an explicit “thinking budget” or “anti-think” system prompt pattern for this model
Since token budget is the limiting factor, the most direct mitigation is to reduce reasoning length:
- system prompt like “Do not output `<think>`; answer directly.”
- or “Keep `<think>` under N tokens and always provide a final answer.”

### 4) Consider adding stop-string tooling for agent loops, not for “final answer”
For OpenClaw/tool-call loops, a stop string such as `</tool_call>` is valuable.
For hiding think:
- stopping at `</think>` would often prevent the final answer from being generated (if the model places the answer after `</think>`), so it can make “empty outputs” worse.

### 5) Apply the same strategy to `runner.py` if you rely on `--hide-think` there
`runner.py` uses the same `ThinkFilter` semantics (`runner.py:342-385`).
It can also print empty output when the generation ends inside an unclosed `<think>`.

## What to test next (minimal)

1) Run a prompt that previously “goes think-only”, with:
   - a stricter system prompt (in `chat.py` use `--system ...`)
   - larger `--max-new-tokens` (e.g. 4096/8192 if feasible)
2) Compare:
   - raw output (without `--hide-think`)
   - stripped output (with `--hide-think`)
3) Check whether the output contains `</think>` before the generation stops.
   - If it does not: the empty outputs are expected from current stripping behavior.
   - If it does: then stripping logic may be removing valid “final” content, and we should re-audit the exact emitted format.
