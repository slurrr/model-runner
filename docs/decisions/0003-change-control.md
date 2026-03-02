# 0003 - Notes-first workflow and explicit approval for code changes

Date: 2026-02-28

## Context
We’re expanding from “quick runner scripts” into a version-controlled repo with docs, notes, and (eventually) integration glue (e.g. OpenClaw tool loops). Unplanned code changes make it harder to reason about behavior across models.

## Decision
- Default workflow is **notes first**: capture model quirks and the hypothesized fixes in `models/<model>/<backend>/notes/` before changing runner scripts.
- Do not make further code changes unless explicitly requested.
- Record higher-level process/structure decisions under `docs/decisions/`.

## Consequences
- Model experiments produce durable documentation even when code stays stable.
- Script changes become deliberate (planned, justified, and easier to review).
- Decisions stay lightweight but auditable as the repo grows.
