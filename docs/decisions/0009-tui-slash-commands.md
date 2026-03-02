# Decision: TUI slash commands are local-only and registry-driven

Date: 2026-03-02

## Context
As the TUI grows (multiple backends, multiple config knobs), it’s easy to lose track of “what is actually set right now” during a session.
We want a simple way to inspect the effective system prompt, prefixes, decoding parameters, UI tick/follow state, and backend identity without leaving the TUI.

## Decision
Add a slash-command system to the TUI with these constraints:

- Slash commands are **local-only**:
  - they are not sent to the model
  - they are not appended to the conversation history
- Commands are implemented via a small **registry** with help metadata (name, aliases, usage, summary).
- Command output is appended to the transcript as an “info message” widget so results are scrollable and part of the session’s visible trace.
- Commands are read-only in v1; future mutations (if any) must be explicit and safe (and blocked during generation).

## Consequences
- Users can verify effective configuration at runtime (`/show`, `/system`, `/prefix`) and reduce “why is it behaving like this?” debugging time.
- The registry makes commands discoverable (`/?`) and keeps implementation organized as commands expand.
- Keeping commands local-only avoids accidental prompt pollution and makes command output safe to include in the transcript.

