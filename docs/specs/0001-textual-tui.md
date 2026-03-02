# Spec: Textual TUI chat with collapsible streaming “thinking”

Date: 2026-02-28

## Goals
- Provide a terminal UI (TUI) for local HF chat that:
  - shows user input in a fixed grey “input band” at the bottom,
  - shows assistant output above in the normal terminal background,
  - renders “thinking” content in grey and final/normal output in white,
  - hides `<think>` / `</think>` markers themselves but preserves the tokens between them,
  - supports a collapsible thinking section:
    - collapsed by default with an animated `> thinking...` header,
    - click to expand → `▼ thinking...` and stream/show the thinking text in grey.
- Keep backend-specific logic separated (do not overload `chat.py`); add a new entrypoint for the TUI.

## Non-goals
- A full agent/tool execution UI (tool parsing/execution can come later).
- Replacing existing scripts (`chat.py`, `runner.py`) immediately.
- A browser UI.

## Recommendation: Textual (built on Rich)
Use **Textual** rather than Rich alone:
- Rich can style and stream text but does not give you a real “click to expand/collapse” UI.
- Textual provides widgets, mouse events, scrolling regions, input handling, timers/animation, and uses Rich for rendering/styling.

Dependency impact:
- Add `textual` (and its dependencies) to `requirements.txt` when implementing.

## Proposed entrypoint
- New script: `tui_chat.py`
  - Keeps `chat.py` as the simple reference CLI.
  - Shares model/tokenizer loading and generation code by importing from a small shared module if desired (optional refactor).

## UX requirements

### Layout
- Top: scrollable transcript area
  - user messages (normal white, prefixed `> ` or `You:`)
  - assistant messages:
    - final answer text in white
    - thinking section in grey, collapsible
- Bottom: fixed input band
  - grey background (full width)
  - input prompt + editable text

### Thinking section behavior
- Default state: collapsed
  - show a single-line header: `> thinking...` (grey text, no thinking content visible)
  - header is animated (see animation spec below)
  - while collapsed: thinking tokens are still captured; optionally count tokens/characters in header (e.g. `> thinking… (1234 chars)`)
- On click (or keybinding fallback):
  - toggle expanded state:
    - expanded header: `▼ thinking...`
    - show streaming thinking text in grey as it arrives
- Hide `<think>` and `</think>` markers themselves (don’t display them), but display the content between them.

### Animation spec (“white band flows through”)
Target: make it obvious the model is working even when output is hidden.
- Animate the header string `> thinking...` by applying a moving highlight over a fixed 3-character window:
  - base color: grey
  - highlight window: white (same characters, just color)
  - window advances by 1 character every ~100–150ms and loops
Example (conceptual coloring):
- `> tHIinking...` where `HIi` is highlighted, then shifts to the right.

Implementation detail:
- In Textual, use a `set_interval()` timer to update a reactive `phase` integer.
- Render the header as a Rich `Text` object with per-span styles.

### Streaming “hung” mitigation
- If the model is inside a long thinking block and the section is collapsed, the header animation still runs so the UI never looks frozen.
- Optional: show an elapsed timer (e.g. `> thinking… 12.4s`).

## Model-aware parsing: routing tokens to “thinking” vs “answer”

### Why parsing is needed
For Nanbeige4.1-3B, `<think>`/`</think>` are not “special tokens” and will appear in decoded text. We want to:
- hide the markers,
- show the content between them in grey (possibly hidden behind the collapsible),
- show the content outside them as the final answer in white.

### Streaming parser requirements
The model output arrives in chunks (string pieces). Markers can be split across chunks (`"<th"` + `"ink>"`), so parsing must be incremental with a small buffer.

Define a state machine:
- `mode = "answer" | "think"`
- known markers:
  - start: `<think>` plus optional alternates if desired (`<|begin_of_thought|>`, etc.)
  - end: `</think>` plus alternates

Algorithm (incremental):
- Maintain `buffer` string.
- On each incoming piece:
  - append to `buffer`
  - while `buffer` contains a start/end marker relevant to current mode:
    - if `mode == "answer"`:
      - find earliest start marker
      - emit text before marker → ANSWER stream (white)
      - drop marker itself
      - switch `mode="think"`
    - if `mode == "think"`:
      - find end marker
      - emit text before marker → THINK stream (grey)
      - drop marker itself
      - switch `mode="answer"`
  - keep a trailing `max_marker_len` tail in `buffer` to catch split markers

Edge cases:
- If generation ends in `mode="think"` with no closing tag:
  - still retain the captured think text
  - UI should not display an “empty assistant message”; instead show:
    - collapsed header remains, and optionally indicates “thinking truncated”
  - this avoids the “empty reply” symptom seen with naive stripping.

### Optional: token-id-based parsing (future)
More robust but more work:
- implement a custom streamer that yields token IDs
- detect marker token IDs directly (`<think>` is a single token ID for Nanbeige)
This avoids split-marker problems and allows exact token counting.

## Textual widget plan

### Transcript area
Use a `VerticalScroll` container holding message widgets:
- `UserMessage` widget:
  - renders `> <text>` in white
- `AssistantMessage` widget:
  - contains:
    - `ThinkingPanel` (collapsible)
    - `AnswerLog` (always visible)

### ThinkingPanel
Components:
- header row (clickable)
  - label text with animation
  - optional timer / char count
- body (hidden unless expanded)
  - a `RichLog`-like widget that appends grey text as it streams

Mouse + keyboard:
- Click header toggles expand/collapse.
- Keybinding fallback (for terminals without mouse):
  - `t` toggles the most recent thinking panel.

### Input band
Use Textual `Input` at bottom in a `Container` styled with:
- grey background across full width
- white cursor / input text (or slightly brighter grey)

### Transcript follow behavior (deterministic contract)
- Follow mode is **intent-driven** (single authority).
- Break follow on explicit upward navigation intent:
  - mouse wheel up
  - `PageUp`
  - `Home`
  - line-up scroll actions
- Resume follow on explicit resume intent:
  - `End` (must jump to bottom and enable follow)
  - downward navigation that reaches bottom
- Avoid passive follow toggling based on inferred `scroll_y` deltas; this causes competing state transitions.

## Concurrency / generation

### Approach
- Keep HF generation in a background thread (as today) to avoid blocking the UI.
- Stream text pieces back to the UI via a thread-safe queue.
- In Textual, use `call_from_thread()` or a periodic poll to drain the queue and update widgets.

### Minimal data model per assistant turn
- `raw_stream`: full raw text (including `<think>` markers) captured for debugging/logging
- `think_stream`: extracted text between markers
- `answer_stream`: extracted text outside markers
- `finished`: boolean
- `ended_in_think`: boolean (if generation finished while still in think mode)

## CLI surface (when implemented)
Keep it close to `chat.py`:
- `python tui_chat.py <model_id> [--config ...] [--max-new-tokens ...] [--temperature ...] ...`
- Add TUI-specific toggles:
  - `--show-thinking` (default false; start expanded)
  - `--no-animate-thinking` (disable header animation)
  - `--save-transcript <path>` (optional)

## Logging and debugging
- Optional debug pane or file output:
  - raw model output including markers
  - extracted streams
- When a turn ends with no answer text, show a visible hint:
  - “(No final answer produced; generation ended during thinking. Increase max_new_tokens or adjust prompt.)”

## Nanbeige4.1-3B specific notes (implementation guidance)
- Prefer to always supply an explicit system message in the prompt to avoid the template’s default Chinese system injection.
- Expect long `<think>` spans; the UI should remain responsive even if no answer text is produced.
- Hiding markers but showing inner tokens is the right default for this model.
