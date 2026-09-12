# Dashboard Repo Architecture Note

Date: 2026-03-15

Status: Scratchpad

Downstream repo name:
- `lab-llm`

Purpose:
- keep repo-facing notes about the downstream dashboard architecture
- document the current expected boundary from `model-runner`
- avoid turning this repo into the dashboard implementation home

This file is intentionally downstream-facing and non-authoritative for the dashboard repo.
It exists so this repo can remember what the downstream architecture expects today.

## Current boundary assumptions
The `lab-llm` dashboard repo is expected to consume `model-runner` through external interfaces only:
- JSON/app telemetry events
- optional Prometheus-compatible metrics

It should not import internal modules from this repo as a primary integration path.
It should also own frontend-facing REST/SSE rather than expecting `model-runner` to provide them.

## Current architectural shape
The dashboard repo is expected to have three layers.

### 1. Ingestion layer
Consumes telemetry from `model-runner`.

Inputs:
- fixture JSONL during early UI development
- live-appended JSONL from `model-runner`
- optional Prometheus scrape data
- optional GPU/system exporter metrics

Later, a more direct ingestion channel may be added if it clearly earns the extra complexity.

Responsibilities:
- normalize external input into dashboard-owned read models
- maintain live session state
- persist recent history as needed

### 2. Dashboard service layer
Serves dashboard-facing queries and live updates.

Likely responsibilities:
- session list/detail APIs
- experiment compare APIs
- live subscription fanout
- retention/downsampling policy
- local persistence management

### 3. Web UI layer
Provides the browser experience.

Likely areas:
- overview
- sessions
- compare
- inspect
- logs
- future chat area in the same shell

## Current entity expectations
The dashboard repo should expect these concepts from `model-runner`:
- session
- experiment
- load report
- turn summary
- runtime sample
- log event
- error event

## Current app/API expectations
The dashboard repo should assume:
- events are append-only
- payloads are versioned
- unknown fields may appear over time
- some metrics are unavailable on some backends

The dashboard should therefore:
- treat missing fields as capability gaps, not hard failures
- clearly distinguish unavailable vs zero
- prefer canonical fields over backend-specific extensions

## Current Prometheus posture
Prometheus is expected to be useful for:
- quick charts
- standard metric collection
- reuse of existing exporter tooling

Prometheus is not expected to be sufficient for:
- turn-level inspection
- knob truth
- logs
- request inspect views
- chat/session semantics

The dashboard repo should therefore treat Prometheus as supplemental, not primary.

## Current integration order
Shortest path for the downstream repo:
- develop against fixture JSONL
- switch to live-appended JSONL from `model-runner`
- add a more direct ingestion channel later only if it materially improves the system

## Current UI posture
The downstream UI should be:
- observability-first initially
- ready to share a shell with future browser chat
- explicit about live vs completed state
- explicit about unknown/unavailable metrics

## Diagnostic expectation
The downstream dashboard should help distinguish model/config issues from fused-kernel/runtime issues.

In particular, for cases like Qwen3.5 where:
- conservative paths such as HF `sdpa` behave correctly
- faster fused paths such as HF `flash_attention_2` or vLLM/FlashInfer may loop or fail

the dashboard should make it possible to compare and inspect:
- attention backend in use
- prefill behavior versus decode behavior
- cache-on versus cache-off runs when available
- throughput versus correctness regressions
- runtime errors and termination mode

The goal is not only “what is faster,” but “which kernel path changes model behavior.”

## Open notes
- If the dashboard repo needs long-retention storage, that should be solved there, not here.
- If the dashboard repo wants richer compare workflows or annotations, that should be solved there, not here.
- If the telemetry schema changes, this note should only summarize the downstream impact, not restate the full schema.
