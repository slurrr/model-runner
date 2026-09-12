# 0017 - Canonical telemetry schema with dual export surfaces

Date: 2026-03-15

## Context
Decision `0016` establishes that this repo owns telemetry production and export, while the browser dashboard lives in a separate repo.

That leaves two related questions:
- what is the canonical telemetry model in this repo?
- how should external consumers read it?

We need:
- one stable internal schema across backends
- one export path for numeric metric tooling
- one richer export path for session- and turn-aware consumers

If we let Prometheus define the model, we lose important semantics like:
- per-turn request truth
- logs
- inspect payloads
- experiment/session structure

If we only expose a custom app API, we lose easy compatibility with Grafana and standard exporter workflows.

## Decision
This repo will use a canonical telemetry schema as the source of truth.

That schema will cover:
- sessions
- load reports
- turn summaries
- runtime samples
- logs
- errors

This repo will support two export surfaces derived from that schema:

1. App export surface
- JSON-shaped events for rich downstream consumers
- MVP delivery in this repo is in-process publication with an optional JSONL sink
- Network-facing app APIs are deferred to downstream consumers unless a later decision changes that

2. Metrics export surface
- Prometheus-compatible numeric metrics for scrape-based tooling

Prometheus is an export format, not the canonical model.
Prometheus support is expected to be disabled by default and enabled explicitly when needed.

The telemetry model should be:
- backend-agnostic by default
- extensible for backend-specific facts
- additive and versioned
- tolerant of unknown/unavailable fields without fabricating values

## Consequences
- Telemetry work in this repo should start from canonical event/entity design, not from metric label design.
- Any Prometheus metric added here should be traceable back to a canonical schema field.
- Rich downstream consumers such as the future dashboard repo should prefer the app export surface.
- The dashboard repo should ingest telemetry and expose its own REST/SSE surfaces instead of relying on `model-runner` to host them.
- Backend adapters must explicitly represent unavailable facts instead of inferring them.
- Future schema evolution should prefer additive changes and explicit versioning.
