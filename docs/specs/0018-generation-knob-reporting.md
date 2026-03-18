#!/usr/bin/env markdown
# Spec: Generation knob reporting (sent / deferred / ignored)

Date: 2026-03-09

Status: Draft

Related umbrella contract:
- `docs/specs/0015-backend-standardization-contracts.md`

## Context / problem
This repo is a “local LLM playground”, and the user expectation is:
- If a knob is exposed, it must either do something or be explicitly reported as ignored/deferred.
- The TUI should show **what we actually sent**, what we **deferred to backend/model defaults**, and what was **ignored**.

Right now, this is inconsistent across backends:
- Some backends silently ignore unsupported knobs.
- Others warn once, but the TUI doesn’t have a consistent per-turn view.

## Goals
- Standardize a per-turn report of generation knobs:
  - `sent`: key/value pairs actually applied to the backend call
  - `deferred`: keys intentionally omitted so backend/model defaults apply
  - `ignored`: keys provided by CLI/config but not supported or not applied
- Expose this report via `/show gen` with a stable, testable output contract.

## Non-goals (v1)
- Perfect detection of whether a server truly used a parameter internally (we can’t always prove it).
- A complete mapping for every possible vLLM/llama.cpp/private knob.
- Reporting UI knobs or prompt-shaping knobs (e.g. `show_thinking`, `assume_think`, `chat_template`, `system`, `/image` state).

## Data model (required)
Extend `TurnRecord` with:
- `knobs: dict[str, object] | None`

Shape (recommended):
```json
{
  "sent": { "temperature": 0.7, "top_p": 0.95 },
  "deferred": ["top_k"],
  "ignored": ["typical_p"],
  "notes": ["GGUF: chat API failed; fell back to plain completion"]
}
```

Rules:
- `sent` MUST only contain values included in the actual backend request (payload / kwargs / settings) for the completed turn.
- `deferred` MUST list knobs the user could set but did not (or were default/None) and thus were intentionally omitted.
- `ignored` MUST list knobs present in CLI/config that were not applied.
- `notes` is OPTIONAL and MUST be omitted when empty.

## Canonical knob names (MUST)
To support cross-backend comparison, the report MUST use canonical repo knob names (user-facing names), not backend payload keys.

Examples:
- Use `stop_strings` (repo knob), not `stop` (OpenAI payload field).
- Use `repetition_penalty` (repo knob), not `repeat_penalty` (llama.cpp kwarg).
- Use `max_new_tokens` (repo knob), not `max_tokens` (OpenAI payload field).

Backend adapters MAY include payload aliases in `notes`, but MUST NOT replace the canonical names in `sent/deferred/ignored`.

## Derivation point (MUST)
The report MUST be derived from the **final** request/settings actually used for the completed turn, after:
- retries
- fallbacks (e.g. GGUF chat → plain completion)
- normalization (e.g. clamping, “shrink max tokens”, dropping oldest turns)

This ensures `/show gen` answers “what happened” rather than “what we hoped would happen”.

## User-set detection (MUST)
`ignored` vs `deferred` MUST be determined using **user-set detection**, not only raw arg values.

Requirement:
- A knob is considered **user-set** if it came from CLI/config/profile/machine overrides (i.e. not a parser default).
- Parser defaults MUST NOT be treated as “user intent”.

Implementation note:
- This repo already tracks config origins; use that to avoid misclassifying defaults.

## `/show gen` output contract (MUST)
Default output:
- `sent.*` block
- `deferred` list
- `ignored` list

Verbose (`/show gen --verbose`):
- includes `args.*` (raw configured values)
- includes `notes` if present

JSON mode (`/show gen --json`):
- returns full `TurnRecord.knobs` object (plus `args` if `--verbose`).

### Before the first completed turn (MUST)
If there is no completed turn yet:
- `/show gen` MUST NOT pretend it has “sent” values from a real turn.
- It MUST either:
  1. print `No completed turn yet.` and exit, OR
  2. show an **intent view** derived from current args/config, clearly labeled as intent (recommended).

If (2) is chosen, the output MUST include:
- `mode: intent`
- the same top-level shape as a real turn (`sent/deferred/ignored`), derived purely from args/config
- and MUST NOT include backend-mutated values (since no request occurred).

## Backend requirements
Each backend MUST populate `TurnRecord.knobs` per turn.

### HF
Derive `sent` from `generate_kwargs` (post-normalization).
Examples:
- `top_k` in `sent` only if non-None and actually passed.
- `stop_strings` included only when supported and enabled.
Non-knobs / backend internals:
- HF always passes various internal fields (e.g. `do_sample`, `pad_token_id`, `eos_token_id`) that are not exposed as repo knobs.
- These MUST NOT be included in canonical `sent/deferred/ignored` unless/until we explicitly promote them to repo knobs.

### GGUF (llama.cpp)
Derive `sent` from `create_chat_completion` / `create_completion` kwargs.
Do not treat “repo does not expose this knob at all” as `deferred` by default.
Only mention non-surfaced knobs in `notes` when it provides concrete user value for troubleshooting or tuning.
If the backend switches paths (chat → plain completion, or max token shrink), it MUST record a `notes[]` entry describing the change.

### EXL2
Derive `sent` from ExLlama settings object used for generation.

### vLLM (managed OpenAI-compatible)
Derive `sent` from the final HTTP payload after filtering.
Additionally:
- Track `deferred` knobs explicitly when we omit them (e.g. `top_k=None`).
- Track `ignored` knobs for parameters we parse but don’t send (or can’t send due to protocol).

## Ordering and stability (MUST)
- `ignored` MUST be sorted by canonical knob name.
- `deferred` MUST be sorted by canonical knob name.
- `sent` keys MUST be emitted in stable sorted order in JSON.

## “Sent even if user didn’t touch it” (MUST)
Some transports always send certain fields (e.g. `temperature`, `top_p`) even if the user never changed them.

Rule:
- If the backend actually sent the field, it MUST appear in `sent` (because it is “what we actually sent”).
- Whether it was user-set or default MAY be reflected in `notes`, but MUST NOT change the definition of `sent`.

## One-time startup summary (optional)
Backends MAY emit a one-time `Meta(ignored_knobs=[...])` at session start or first turn,
but this does not replace per-turn `TurnRecord.knobs`.

## Testing checklist
- Set a knob that is unsupported on a backend (e.g. `min_p` on HF):
  - verify `/show gen` shows it in `ignored`.
- Omit a knob (leave None):
  - verify `/show gen` shows it in `deferred`.
- Set a knob and ensure it is in the actual request:
  - verify `/show gen` includes it in `sent`.
