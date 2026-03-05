# ds32b-alex (Ollama)

## Where
- Backend: Ollama (`/api/chat`)
- Ollama model name: `ds32b-alex`
  - In this repo, use: `ollama:ds32b-alex`
- Repo config: `models/ds32b-alex/ollama/config/config.json`
- Ollama template snapshots (for debugging prompt/format issues):
  - Active at build time (from Modelfile): `models/ds32b-alex/ollama/templates/current.jinja`
  - Reference (original tokenizer/HF-style chat template): `models/ds32b-alex/ollama/templates/ds32b.jinja`

## TL;DR
- This model often emits “thinking” **without** an opening `<think>` tag and later emits `</think>` as a delimiter.
  - For this model, `assume_think=true` is usually required so the UI routes early text to the thinking panel until the first `</think>`.
- Some `</think>` markers can still appear in the **answer** panel due to how Ollama streaming sometimes splits “thinking” into `message.thinking` while leaving `</think>` inside `message.content`.
- Formatting weirdness like `Alex: Alex: ...` and `***Alex***` is almost certainly **template-driven** because the active Ollama template hardcodes `Alex:` in the assistant role prefix.

## Recommended commands (this repo)

Unified TUI (recommended):
```bash
tui ollama:ds32b-alex
tui ollama:ds32b-alex --config ds32b-alex
```

If you haven’t installed the console entrypoint:
```bash
python tui.py ollama:ds32b-alex
python tui.py --config ds32b-alex
```

Backend-only sanity check (no TUI):
```bash
python ollama_chat.py ds32b-alex
```

## Current repo defaults for this model
From `models/ds32b-alex/ollama/config/config.json` (as of 2026-03-02):
- `stream=true`
- `max_new_tokens=2048`
- `ollama_think=false`
- `assume_think=true`
- `temperature=0.7`, `top_p=1.0`, `top_k=null`
- UI: `scroll_lines=9`, `ui_tick_ms=33`, `ui_max_events_per_tick=33`

## Thinking / delimiter behavior

### Observed behavior
- Typical pattern:
  - starts with reasoning text (no `<think>` marker)
  - later emits `</think>`
  - then emits final answer text

### How the TUI routes thinking for this model
- With `assume_think=true`, the router starts each assistant turn in **think** mode and switches to **answer** mode at the first end marker (`</think>` or known variants).
- This is “optimistic streaming”: early tokens are shown as thinking immediately, assuming `</think>` will arrive.

### Why `</think>` can still show in the answer panel (known gotchas)
This repo’s Ollama backend handles two different “streams”:
- If Ollama provides `message.thinking`, it is emitted directly to the thinking panel.
- If `message.thinking` is present, the current implementation emits `message.content` directly as answer (it does not run the content through the router).

If Ollama’s `message.content` contains a literal `</think>`, you’ll see the tag show up in the answer panel.

Practical debugging steps:
- Enable transcript capture and inspect a “bad” turn:
  ```bash
  tui ollama:ds32b-alex --save-transcript transcripts/ds32b-alex.jsonl
  ```
  Look at the JSONL record:
  - if `answer` contains `</think>` while `think` is non-empty, the delimiter likely came from `message.content` while `message.thinking` was also present.

## Prompting / templates (high impact for this model)

### Active Ollama template (build-time)
`models/ds32b-alex/ollama/templates/current.jinja` currently:
- emits `<｜Assistant｜>Alex: ...` for assistant turns (hardcoded speaker prefix)
- injects a “nice and concise” prefix into each user message

Implications:
- If the model also learns to self-label with `Alex:` you can get duplicated prefixes: `Alex: Alex: ...`.
- You may also see alternative self-labeling like `***Alex*** ...` as a learned variation of the same pattern.

### Reference template (tokenizer/HF-style)
`models/ds32b-alex/ollama/templates/ds32b.jinja` shows a more standard structure and includes logic to drop earlier reasoning when replaying assistant turns:
- it strips content before `</think>` when building the next prompt (reduces reasoning-tag pollution in history)

### Template experiments worth trying
If formatting is a priority:
- Remove the literal `Alex:` from the assistant prefix in the Ollama template.
- Move “nice and concise” into the system prompt (once) rather than injecting it into every user message.
- Consider adopting the “strip-before-</think>” history behavior from the reference template.

## Known issues / symptoms (repo observations)
- Symptom: `</think>` sometimes appears in the answer panel even with `assume_think=true`.
  - Likely cause: Ollama streaming provides `message.thinking` (handled as think) but leaves the delimiter in `message.content` (handled as answer without routing).
  - Debug: save JSONL and confirm whether `answer` contains the delimiter while `think` is also present.
- Symptom: inconsistent speaker labeling:
  - `Alex: output` (expected)
  - `Alex: Alex: output` (duplicated prefix)
  - `***Alex*** output` (variant prefix)
  - Likely cause: template hardcodes `Alex:` and the model sometimes emits its own labels.

## What to capture when debugging
When something looks “off”, capture:
- `--save-transcript ...` JSONL record (raw/think/answer + timing)
- The exact Ollama model name used and whether it was invoked as `ollama:<name>`
- The active Ollama template content (`current.jinja`) and any recent changes to it
- Whether the run produced `message.thinking` fields (if you add a one-off debug print later, that’s the key signal for delimiter leakage)

## Best use cases (in this repo)
- Interactive TUI sessions where you want to see/hide reasoning quickly.
- Template experimentation (this model is sensitive to role/prefix scaffolding).

## Not ideal for (current state)
- Clean “assistant-only” transcripts without any speaker labels or delimiters (until template is adjusted and delimiter handling is tightened).
