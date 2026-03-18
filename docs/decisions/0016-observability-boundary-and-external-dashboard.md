# 0016 - Keep telemetry in this repo and build the dashboard separately

Date: 2026-03-15

## Context
This repo is a local model lab for backend experimentation, runtime tuning, and model/application research.

It already spans:
- multiple inference backends
- model-first config and notes
- terminal interaction surfaces
- backend standardization work

Observability is aligned with the repo's purpose, but a full browser dashboard introduces a second product with distinct concerns:
- frontend architecture
- dashboard persistence and compare workflows
- visualization UX
- external adoption by users who may not want the full runner repo

Keeping all of that in this repo would increase clutter and weaken separation between:
- runner/backend experimentation
- dashboard product development

## Decision
This repo will own the observability production side:
- canonical telemetry schema
- backend telemetry adapters
- session and turn event emission
- optional export endpoints such as Prometheus-compatible metrics
- documentation for external observability consumers

This repo will not own the long-term browser dashboard product.

The dashboard should live in a separate repo and consume this repo through stable external interfaces rather than importing internal runner modules directly.

We will support two external boundaries:

1. Metrics boundary
- Prometheus-compatible numeric metrics export

2. App boundary
- richer JSON/event interfaces for sessions, turns, logs, inspect data, and experiment metadata

Prometheus support is optional but recommended as an export format.
It is not the canonical internal model.

## Consequences
- Observability work in this repo should focus on schema, emission, and export.
- Browser dashboard implementation should not be added to this repo as the default path.
- Prometheus metrics must be derived from the repo-owned telemetry model, not used as the primary source of truth.
- A downstream dashboard can be useful both to this repo and to other experimenters with different runners if the boundary stays clean.
- Future implementation specs in this repo should describe telemetry contracts and export behavior, while dashboard UI specs belong in the dashboard repo.
