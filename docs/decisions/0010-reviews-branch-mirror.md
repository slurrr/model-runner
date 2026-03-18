# 0010 - Mirror PR review output into a `reviews` branch

Date: 2026-03-04

## Context
- PR review bots (e.g. `codex-connector`) produce useful findings on GitHub.
- Local Codex sessions in this repo may not reliably access GitHub APIs, and copy/pasting review output into local sessions is tedious.

## Decision
- Add a CI workflow that mirrors bot review output into a dedicated git branch named `reviews`.
- `reviews` is an orphan branch that contains only generated artifacts under `docs/reviews/`.
- CI is the only writer to `reviews`.
- Do not publish mirror artifacts for PRs from forks.

## Consequences
- Pros:
  - Local/offline tooling can fetch and read review bundles via `git show`, no checkout required.
  - Review history is versioned and searchable.
- Cons:
  - The `reviews` branch is generated content and should not be edited manually.
  - Fork PRs will not publish review bundles without a separate `pull_request_target` design (intentionally deferred).

