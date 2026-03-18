# Here are the main “clever” strategies that fit this repo:

## 1) Sliding “working set” + summarization (classic)

Policy:

- Keep the last N turns verbatim (your working set).
- Everything older collapses into a running summary (“long-term memory”) that gets injected as a system/developer message.

Practical details that matter:

- Summaries must be stable and structured (facts, decisions, TODOs), not prose.
- Update summary only when needed (e.g. every 3 turns or when tokens exceed threshold) to avoid drift/cost.

Where it plugs in here:

- Right before backend prompt rendering (i.e., build trimmed_messages).
- This is exactly the place we’re already standardizing with trimmed_messages + context trimming.

## 2) Hybrid: “facts memory” + “scratchpad memory”

Instead of one summary, keep two:

- Facts/Preferences/Config: stable, rarely changes.
- Scratchpad/Progress: current task state, recent steps.

This reduces “summary churn” and keeps the model anchored.

## 3) Retrieval-lite (no vector DB) using “notebook pages”

Create a local “notebook” file per session:

- Append key items (decisions, code refs, constraints).
- When tokens get tight, inject only the relevant notebook sections (by simple keyword match).

This avoids embeddings infra and is surprisingly effective for coding/ops workflows.

## 4) Tool-augmented memory (once tool harness exists)


- takes the overflow transcript chunk
- returns a structured summary block
Then store it and inject it next turn.

This gives you controllable, logged summarization, and you can swap the summarizer model later (fast small model).

## 5) “Never re-feed chain-of-thought” (already aligned with your direction)

This is basically history_strip_think=true (spec 0019):

- show thinking in UI
- don’t re-inject it into context
This alone can reclaim a lot of budget on reasoning-ish models.

## 6) “Token budget governor”

Before each turn:

- compute prompt_tokens (we’re standardizing this now)
- if near limit:
    - drop low-value messages first (banter, acknowledgements)
    - then summarize
    - then drop entire tool bundles as units (per spec 0020)
    This makes behavior predictable and debuggable.