#!/usr/bin/env markdown
# Spec: Local observability dashboard

Date: 2026-03-15

Status: Draft

Related docs:
- `docs/observability_dashboard`
- `docs/specs/0015-backend-standardization-contracts.md`
- `docs/specs/0016-token-accounting-and-real-toks.md`
- `docs/specs/0017-tui-logging-ring-buffer.md`
- `docs/specs/0018-generation-knob-reporting.md`

## Context / problem
This repo is no longer just a terminal chat app. It is a local model lab that needs:
- runtime truth
- backend comparison
- experiment tracking
- a browser-friendly surface for tuning and inspection

The current TUI is useful, but it is not the right long-term surface for:
- long-running metric visualization
- side-by-side backend comparison
- hardware and cache monitoring
- cross-run experiment review

The existing `docs/observability_dashboard` direction is correct on intent, but it is framed mostly as infra:
- Prometheus
- Grafana
- exporters

That is necessary, but not sufficient.

What this repo actually needs is a repo-owned observability product with:
- a shared runtime telemetry model
- a local web dashboard
- a clean path to later add web chat on the same surface
- optional Prometheus/Grafana interoperability, not product ownership outsourced to them

## Goals
- Add a local browser dashboard for runtime observability and experiment comparison.
- Preserve the TUI as a valid control surface during rollout.
- Keep the first-class data model backend-agnostic and repo-owned.
- Expose runtime truth, not inferred summaries.
- Make it possible for a future web chat UI to live in the same app shell as observability.
- Support both live monitoring and short-run historical comparison.
- Keep the system local-first and usable without cloud services.

## Non-goals (v1)
- Replacing the TUI as the primary control surface immediately.
- Building a generic MLOps platform.
- Full distributed tracing or multi-host infra management.
- Perfect parity for every engine-specific metric on day one.
- A polished annotation-heavy experiment notebook system.

## Product direction
We should build this as a single local web app with two long-term product areas:
- `Observe`: runtime, hardware, cache, latency, throughput, logs, request truth
- `Chat`: future browser chat/control UI

V1 only delivers `Observe`, but the app shell and server boundaries should assume `Chat` will be added later.

This means:
- one local server process for browser UX
- one shared session/experiment model
- one navigation shell
- one event stream transport for live updates

Do not start with Grafana as the main user experience.
Grafana is useful for external dashboards and quick time-series panels, but it should consume the repo's telemetry model rather than define it.

## User stories
- As a user, I can see whether a model loaded the way I think it loaded.
- As a user, I can compare HF vs GGUF vs EXL2 vs vLLM on the same model and prompt pattern.
- As a user, I can see tok/s, TTFT, decode speed, memory use, cache growth, and batching behavior while a run is active.
- As a user, I can inspect the exact generation/request state that was applied for a turn.
- As a user, I can review a completed run later without scraping terminal output.
- As a user, I can keep using the TUI while the dashboard passively observes.
- As a user, I can later open a chat pane in the same browser app without switching to a different product.

## Principles
- Repo-owned schema first.
- Backend truth over UI convenience.
- Local-first operation.
- Same concepts across TUI and web.
- Keep engine-specific facts available without polluting the cross-backend baseline.
- Prefer append-only event capture plus derived summaries over mutable ad hoc state.

## Proposed architecture
Build the system in four layers.

### 1. Runtime telemetry contract
Add a shared telemetry contract emitted by all backend sessions.

This contract should include:
- session identity
- experiment identity
- backend identity
- model/load identity
- per-turn lifecycle events
- periodic runtime samples
- derived summaries

There are already adjacent contracts in the repo:
- turn events
- token accounting
- log ring buffer
- generation knob reporting

The new observability work should extend those, not replace them.

### 2. Local observability service
Add a small local service in-repo that:
- receives telemetry from active sessions
- stores recent history locally
- serves a browser UI
- exposes an optional `/metrics` export for Prometheus scraping

This service owns:
- live event fanout
- session registry
- experiment summaries
- retention rules
- dashboard API

It should not own model execution in v1.
The TUI and existing runners remain the control path and publish telemetry into it.

### 3. Browser dashboard
Add a repo-owned web UI focused on:
- live runtime monitoring
- backend comparison
- run drill-down
- hardware and cache inspection
- logs and request truth

The UI should be dashboard-first now, with room for a later chat workspace in the same shell.

### 4. Optional Prometheus/Grafana bridge
Expose Prometheus-compatible metrics from the local observability service.

Use this for:
- external scrape-based dashboards
- ad hoc Grafana panels
- longer retention if the user wants it

This is an integration layer, not the primary internal contract.

## Canonical entities
The dashboard should be built around these entities.

### Session
A live or completed attachment to one backend/model runtime.

Fields:
- `session_id`
- `started_at`
- `ended_at`
- `status`
- `backend_name`
- `transport_name`
- `resolved_model_id`
- `config_path`
- `profile_name`

### Experiment
A logical run grouping one or more sessions for comparison.

Fields:
- `experiment_id`
- `label`
- `notes`
- `created_at`
- `tags`

Examples:
- same model, different backends
- same backend, different quantization
- same prompt, different context length

### Load profile
How a model/runtime actually loaded.

Fields:
- `engine_name`
- `engine_version`
- `model_identifier`
- `model_parameter_count`
- `model_weight_bytes`
- `model_dtype`
- `kv_cache_dtype`
- `quantization_type`
- `context_length`
- `attention_backend`
- `tensor_parallelism`

### Turn
A single user request and its generation outcome.

Fields:
- `turn_id`
- `started_at`
- `ended_at`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `time_to_first_token_seconds`
- `decode_latency_seconds`
- `request_latency_seconds`
- `stop_reason`
- `streaming_enabled`
- `knobs`

### Runtime sample
A periodic point-in-time sample during a live session.

Fields:
- `ts`
- `gpu`
- `memory`
- `kv_cache`
- `batch`
- `throughput`
- `requests`

## Metric groups
The baseline schema should cover these groups.

### Load and configuration
- model identity
- parameter count
- weight bytes
- dtype
- quantization
- context length
- engine and version

### Hardware
- GPU VRAM total/used/free/reserved
- GPU utilization
- memory bandwidth when available
- power usage when available
- CPU and RAM as secondary system metrics

### Runtime allocator
- PyTorch allocated/reserved/max allocated
- allocator fragmentation indicators when knowable
- active tensor count only when cheap and reliable

### KV cache
- total bytes
- used bytes
- utilization ratio
- block size when relevant
- active sequence count

### Throughput
- prompt tok/s
- decode tok/s
- effective tok/s
- tokens generated total
- requests completed total
- requests in flight
- requests queued

### Latency
- TTFT
- prefill latency
- decode latency
- request latency
- percentile summaries for completed turns

### Batch
- batch size current
- batch size average
- batch tokens current
- batch tokens average

### Generation truth
- requested max new tokens
- output tokens
- stop reason
- streaming enabled
- sent/deferred/ignored knobs

## Event model
Use append-only events plus periodic samples.

Required event families:
- `session_started`
- `session_updated`
- `session_finished`
- `load_reported`
- `turn_started`
- `turn_stream_chunk` optional for live counters only
- `turn_metric_reported`
- `turn_finished`
- `runtime_sample`
- `log_line`
- `error_reported`

Notes:
- We already have text streaming events for TUI rendering. Those should remain.
- The observability path should consume summarized telemetry, not store every answer token forever by default.
- If streaming text is later shown in the web chat UI, it can subscribe to the existing text event stream separately.

## Integration with existing repo
### TUI
The TUI remains usable and keeps its existing event contract.

Required additions:
- publish session lifecycle to the observability service
- publish completed-turn metrics and knob reports
- publish periodic runtime samples while a session is active
- publish log lines or log-tail snapshots using the shared logger

### Backends
Each backend adds a telemetry adapter that maps backend-native facts into the canonical schema.

Backends:
- HF / Transformers / PyTorch
- GGUF / llama.cpp
- EXL2
- Ollama
- managed vLLM
- generic OpenAI-compatible targets where limited facts are available

Rules:
- if a metric is not knowable, report it as unavailable rather than fabricating it
- keep backend-specific extra fields in a namespaced extension area

### Existing specs
This spec depends on and should reuse:
- token accounting from `0016`
- knob truth from `0018`
- logging contract from `0017`
- backend consistency rules from `0015`

## Web app shape
Use a single local web app with a persistent shell.

Recommended top-level areas:
- `Overview`
- `Sessions`
- `Compare`
- `Experiments`
- `Logs`
- `Inspect`
- `Chat` reserved for later

### Overview
Shows active sessions, machine health, and recent experiments.

### Sessions
Shows live and completed sessions with key metrics and status.

### Session detail
Shows:
- model/load summary
- live throughput and latency charts
- GPU and memory charts
- KV cache charts
- batch and request state
- turn list
- logs
- last request / generation truth

### Compare
Lets the user overlay multiple sessions or experiments.

Primary comparisons:
- tok/s
- TTFT
- decode latency
- VRAM use
- cache growth
- stop reasons

### Inspect
Structured detail for:
- effective config
- sent/deferred/ignored knobs
- backend-reported metadata
- raw extension metrics when needed

## Same-surface requirement for future chat
To keep a later web chat UI in the same product surface:
- use one app shell, not a separate dashboard app
- use one session model shared by observe and chat
- use one live event transport
- keep answer/thinking stream support available even if v1 observe does not render full chat

This avoids building two incompatible browser products.

## Storage and retention
V1 should keep storage simple and local.

Recommended:
- in-memory live state
- local SQLite for session, experiment, turn, and aggregate metric persistence
- bounded retention for high-frequency samples

Retention guidance:
- keep full session metadata and turn summaries
- downsample old runtime samples
- keep logs with size limits
- do not persist every token delta by default

## APIs and transports
The local observability service should expose:
- REST endpoints for session and experiment summaries
- a live event stream for runtime updates
- a Prometheus-compatible `/metrics` endpoint

Recommended transport split:
- WebSocket or SSE for live dashboard updates
- REST for list/detail pages
- Prometheus text format for scrape integration

## Prometheus / Grafana posture
Support them, but do not let them define the product.

Prometheus is useful for:
- scrape-based capture
- standard exporter integration
- quick compatibility with existing tooling

Grafana is useful for:
- ad hoc charts
- user-custom dashboards
- long-retention time-series review

But the repo-owned dashboard should be the primary UX because it can show:
- session semantics
- turn semantics
- request truth
- backend-specific drill-down
- future chat integration

Those concepts do not map cleanly to Grafana alone.

## Implementation plan
### Phase 1: telemetry contract
- Define shared telemetry dataclasses/schemas.
- Add a publisher interface that sessions can write to.
- Publish session start/finish, load summary, completed-turn metrics, knob truth, and logs.

### Phase 2: local observability service
- Add a lightweight local service and session registry.
- Add local persistence for sessions, turns, experiments, and sampled metrics.
- Add `/metrics` export derived from the canonical schema.

### Phase 3: browser dashboard MVP
- Build a local web UI with:
  - active sessions list
  - session detail
  - throughput/latency/GPU/KV charts
  - logs
  - inspect panels for load and generation truth

### Phase 4: comparison workflow
- Add compare views and experiment grouping.
- Add labels, tags, and run notes.
- Add baseline and delta views across selected sessions.

### Phase 5: same-surface chat
- Add browser chat using the same session/event model.
- Keep observability panels docked or switchable within the same shell.

## Implementation choices
### Backend/service implementation
Prefer straightforward Python in-repo.

Recommended server shape:
- small Python web server
- thin telemetry ingestion layer
- SQLite-backed persistence

### Frontend
The frontend should be intentional and dashboard-native, not a generic admin panel.

Requirements:
- good desktop layout for charts and comparison
- usable mobile fallback
- visual distinction between live state, completed summaries, and unavailable metrics

Framework choice is open, but it should optimize for:
- fast local iteration
- charts
- live updates
- easy co-location with the Python service

## Risks
- Overfitting the first schema to one backend.
- Letting Prometheus labels become the canonical data model.
- Capturing too much high-frequency data and creating local overhead.
- Building a browser dashboard that cannot later host chat cleanly.
- Hiding unavailable metrics instead of making capability gaps explicit.

## Open questions
- Whether the observability service should auto-start from the TUI or run as a separate helper process.
- Whether experiment grouping is user-created, auto-derived, or both.
- Whether Ollama and generic OpenAI-compatible targets should expose reduced-capability cards with explicit unknowns.
- How much raw request payload history we should persist by default.

## Acceptance criteria
- A user can run an existing session from the TUI and see it appear live in a browser dashboard.
- A session detail page shows load truth, throughput, latency, logs, and generation truth.
- At least HF, GGUF, EXL2, and managed vLLM can populate the baseline schema with explicit unavailable fields where needed.
- Completed sessions are reviewable later without relying on terminal scrollback.
- The local service exposes a Prometheus-compatible `/metrics` endpoint.
- The web app architecture leaves a clean path for a future browser chat UI in the same shell.
