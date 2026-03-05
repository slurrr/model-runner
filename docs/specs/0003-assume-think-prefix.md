# Spec: `assume_think` for models that only emit `</think>`

Date: 2026-03-02

## Problem
Some models (notably certain GGUF checkpoints) emit hidden reasoning **without** an opening `<think>` tag, then later emit a closing `</think>` tag and proceed with the final answer.

With the current “explicit start/end tag” routing, this yields:
- no start marker → router never enters think mode
- all early tokens render as answer (white)
- `</think>` appears late and doesn’t help routing the already-streamed text

## Goal
Add one minimal, opt-in setting:
- `assume_think=true` (default: false)

When enabled for a model, the router starts each assistant turn in “thinking mode” and switches to “answer mode” once it sees an end marker such as `</think>`.

No reclassification fallback is required for MVP (assume the model reliably emits `</think>`).

## Behavior (when `assume_think=true`)
- At assistant turn start:
  - set router mode to `think`
- Streaming:
  - route all text to “thinking” until the first end marker is encountered
  - drop the end marker token(s) from output
  - switch router mode to `answer`
  - route subsequent text to “answer”
- Explicit start markers (if they appear anyway):
  - MUST be dropped from output (never render literal `<think>` / `<|begin_of_thought|>` text)
  - they do not change behavior while already in assumed-think mode

If no end marker appears:
- everything remains routed as thinking for that turn (no fallback in this version).

## Implementation (code)

### Router change (single flag)
Extend the existing incremental think router (e.g. `StreamingThinkParser` / shared `ThinkRouter`) to accept:
- `assume_think: bool = False`

Implementation sketch:
- Add `self.assume_think` field
- Ensure initial mode is set from the flag (constructor or reset path):
  - `self.mode = "think" if assume_think else "answer"`
  - clear buffers
- In `feed()`:
  - if `mode == "think"`, look for end markers (`</think>`, `<|end_of_thought|>`, etc.) as usual
  - emit think text up to the end marker, drop marker, switch to answer
  - do not require a start marker in this mode
  - while in think mode, drop any start markers that appear (do not emit them as text)

### Where to apply (scope)
This is a model-specific behavior knob and may only be used for a single model.

MVP wiring recommendation:
- Implement in the shared router so it’s available everywhere,
- but only **enable it** via config/flags for the specific backend/model that needs it (typically GGUF/Ollama).

HF note:
- HF streaming has a separate “thinking token” counter (`TokenCountingTextIteratorStreamer`) with its own mode tracking.
- If `assume_think` is enabled for an HF model, the UI’s “thinking token” counts will be wrong unless the streamer is updated to start in think mode too.

### Marker set
Reuse the existing end markers already supported in the repo:
- `</think>`
- `<|end_of_thought|>` and known variants (including fullwidth bar versions)

## CLI + config plumbing

### CLI flag
Add a single boolean flag in the unified TUI entrypoint:
- `--assume-think`

Default: false.

Precedence for booleans:
- If you want CLI to override config in both directions, also add:
  - `--no-assume-think`
- Recommended implementation pattern:
  - CLI arg default = `None`
  - if CLI is `True/False`, it overrides config
  - else use config value (or default false)

### Config key
Allow the same setting via JSON config:
- `"assume_think": true`

Resolution:
- CLI flag overrides config (see precedence note above)
- Backend defaults remain false when absent

### Scope
This is a **routing/UI** behavior knob, not a model generation knob:
- it affects how output is displayed/split into thinking vs answer
- it does not change sampling, speed, or the model’s internal reasoning behavior

## Notes / caveats
- This is an “optimistic streaming” approach: early output is displayed as thinking immediately, assuming the model will later produce an end marker.
- It is appropriate only when you have confirmed the model consistently emits `</think>` (or another supported end marker).
- If no end marker appears, you should expect the current UX where the “answer” may remain empty and the turn is treated as “thinking-only”.
