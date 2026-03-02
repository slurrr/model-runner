# 0008 - TUI follow mode is intent-driven

Date: 2026-03-02

## Context
The Textual TUI streams long thinking/answer output while users scroll the transcript. Prior implementations mixed:
- explicit input intent (`PageUp`, `End`, mouse wheel), and
- inferred viewport movement (`scroll_y` delta checks each tick).

This produced competing follow state transitions and race-like behavior where follow would re-enable or disable unexpectedly.

## Decision
Use a single intent-driven state machine for transcript follow behavior.

- Break follow only on explicit upward navigation intent:
  - mouse wheel up
  - `PageUp`
  - `Home`
  - line-up scroll actions
- Resume follow only on explicit resume intent:
  - `End`
  - downward navigation that reaches transcript bottom
- Do not use passive `scroll_y` delta inference to change follow state.
- Route keyboard/mouse scroll actions through one transcript control path so all inputs hit the same follow logic.

## Consequences
- Behavior is deterministic and matches user expectations:
  - scroll up => follow off
  - `End` => jump bottom + follow on
  - scroll down to bottom => follow on
- Fewer timing/layout edge cases from async refresh/mount churn.
- Slightly more explicit input handling code, but easier to reason about and debug.
