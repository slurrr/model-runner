#!/usr/bin/env markdown
# Spec: Telemetry MVP schema, publisher, and export

Date: 2026-03-15

Status: Draft

Related decisions:
- `docs/decisions/0015-token-accounting-and-metrics.md`
- `docs/decisions/0016-observability-boundary-and-external-dashboard.md`
- `docs/decisions/0017-canonical-telemetry-schema-and-dual-export.md`

Related specs:
- `docs/specs/0015-backend-standardization-contracts.md`
- `docs/specs/0016-token-accounting-and-real-toks.md`
- `docs/specs/0017-tui-logging-ring-buffer.md`
- `docs/specs/0018-generation-knob-reporting.md`

## Context / problem
This repo needs a standardized observability layer, but after decisions `0016` and `0017`, its scope is:
- telemetry production
- schema standardization
- backend emission
- optional external export

This repo should not absorb the browser dashboard product.

That means the implementation work here is to define and emit a stable telemetry contract that external consumers can use.

## Goals
- Define an MVP canonical telemetry schema for this repo.
- Add a shared publisher interface usable by existing runners and backend sessions.
- Emit enough telemetry to support:
  - local runtime truth
  - backend comparison
  - external dashboard consumption
  - optional Prometheus/Grafana integration
- Keep the implementation lightweight and local-first.

## Non-goals
- Building the dashboard UI in this repo.
- Designing long-retention analytics storage.
- Capturing every streamed token forever.
- Full parity for every backend-specific metric on day one.

## Deliverables
- shared telemetry schema module
- shared publisher interface
- no-op publisher default
- session-owned telemetry emission from active runners/backends
- optional export surface(s)
- minimal tests for schema and emission behavior

## Event envelope
Every exported telemetry event should use a common envelope.

Required fields:
- `schema_version`
- `event_type`
- `event_id`
- `ts`
- `session_id`
- `payload`

Optional fields:
- `experiment_id`
- `backend_name`
- `resolved_model_id`
- `resolved_model_id_kind`
- `model_display_name`
- `model_display_name_source`
- `model_path`

Notes:
- `schema_version` should be a short string such as `v1`
- `event_type` is one of the event families defined below
- `event_id` must be globally unique enough for append-only consumers
- `payload` contains the event-specific entity body
- the envelope exists for JSON/app exports; Prometheus metrics remain derived numeric views

## Canonical entities
### Session
Required fields:
- `session_id`
- `started_at`
- `ended_at`
- `status`
- `backend_name`
- `transport_name`
- `resolved_model_id`
- `config_path`
- `profile_name`

Optional identity clarification fields:
- `resolved_model_id_kind`
- `model_display_name`
- `model_display_name_source`
- `model_path`

Allowed status values:
- `running`
- `finished`
- `error`

### Load report
Load reports may be emitted partially and incrementally as confirmed facts become available.
Multiple `load_reported` events per session are allowed.
Consumers should merge load reports by `session_id`, carrying forward the latest known value for each field.
Later confirmed values are authoritative for the fields they provide.

Required fields when knowable:
- `engine_name`
- `engine_version`
- `model_identifier`
- `model_parameter_count`
- `model_weight_bytes`
- `model_dtype`
- `kv_cache_dtype`
- `quantization_type`
- `context_length`

Optional fields:
- `attention_backend`
- `tensor_parallelism`
- `model_identifier_kind`
- `model_display_name`
- `model_path`
- `extension`

Recommended `extension` content when requested intent differs from confirmed runtime truth:
- `runtime_truth.requested`
- `runtime_truth.confirmed`
- `runtime_truth.mismatches`
- `runtime_truth.unconfirmed_requested`

### Turn summary
Required fields:
- `session_id`
- `turn_id`
- `started_at`
- `ended_at`
- `streaming_enabled`

Token fields when knowable:
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`

Timing and outcome fields when knowable:
- `time_to_first_token_seconds`
- `decode_latency_seconds`
- `request_latency_seconds`
- `stop_reason`

Timing rules:
- emit `request_latency_seconds` whenever end-to-end elapsed time is known
- emit `time_to_first_token_seconds` when the runner can observe the first streamed token
- emit `decode_latency_seconds` only when it can be derived honestly; do not reuse total request latency as a fake decode latency

Required integration:
- `knobs` should reuse the contract from `0018`

### Runtime sample
Required fields:
- `session_id`
- `ts`

Sampling is session-owned and must not block generation threads.
Publishers may drop or coalesce samples under pressure.

Optional groups:
- `gpu`
- `memory`
- `kv_cache`
- `throughput`
- `requests`
- `batch`
- `extension`

Recommended numeric fields inside groups:

`gpu`
- `vram_total_bytes`
- `vram_used_bytes`
- `vram_reserved_bytes`
- `vram_free_bytes`
- `utilization_percent`
- `memory_bandwidth_percent`
- `power_usage_watts`
- `temperature_celsius`
- `measurement_state`
- `unavailable_reason`
- `diagnosis_state`
- `diagnosis_reason`

`memory`
- `torch_cuda_memory_allocated_bytes`
- `torch_cuda_memory_reserved_bytes`
- `torch_cuda_max_allocated_bytes`
- `torch_cuda_active_tensors`

`kv_cache`
- `total_bytes`
- `used_bytes`
- `utilization_ratio`
- `block_size`
- `sequence_count`
- `measurement_state`
- `unavailable_reason`

`throughput`
- `tokens_generated_total`
- `tokens_generated_per_second`
- `effective_tokens_per_second`
- `reported_speed_type`
- `effective_measurement_state`
- `effective_unavailable_reason`
- `measurement_state`
- `measurement_basis`
- `sample_window_seconds`
- `trust_level`
- `trust_reason`

`requests`
- `requests_completed_total`
- `requests_in_flight`
- `requests_queued`
- `activity_state`

`batch`
- `batch_size_current`
- `batch_size_average`
- `batch_tokens_current`
- `batch_tokens_average`

### Log event
Required fields:
- `session_id`
- `ts`
- `source`
- `message`

This should reuse the formatting and redaction rules from `0017`.
`message` must be the canonical message body, not a preformatted line that repeats timestamp or source metadata already carried by the event.
MVP export is new log records only after telemetry is attached; no historical ring-buffer replay.

### Error event
Required fields:
- `session_id`
- `ts`
- `scope`
- `message`

## Event families
The publisher must support these event families:
- `session_started`
- `load_reported`
- `turn_finished`
- `runtime_sample`
- `log_recorded`
- `error_reported`
- `session_finished`

V1 deliberately does not require per-token stream capture in telemetry storage.

## Event payload rules
- `session_started` payload should contain the canonical Session shape with `status=running`
- `load_reported` payload should contain the canonical Load report shape
- `turn_finished` payload should contain the canonical Turn summary shape
- `runtime_sample` payload should contain the canonical Runtime sample shape
  - Runtime samples should distinguish open-but-idle sessions from active generation with an explicit activity field.
  - Runtime throughput fields must not imply a live tok/s number when no request is in flight.
  - If KV cache runtime usage is unavailable, emit an explicit unavailable state instead of implying a working cache metric.
  - If live throughput is derived from local streamed token counts plus elapsed time, mark it as provisional.
  - Do not fabricate `effective_tokens_per_second` when only raw streamed completion throughput is known.
- `log_recorded` payload should contain the canonical Log event shape
  - `message` should not duplicate `ts` or `source` formatting already present in the payload.
- `error_reported` payload should contain the canonical Error event shape
- `session_finished` payload should contain the canonical Session shape with terminal status
- `session_finished` may use `status=error` for terminal failures
- when a failure is terminal and details are available, emit `error_reported` before `session_finished`

Each payload may include:
- `extension`: backend-specific extra fields

Rules for `extension`:
- it must be namespaced by backend or subsystem when practical
- consumers must treat it as optional
- canonical fields must not be hidden only in `extension`
- example: `{ "vllm": { "scheduler": "fcfs" } }`

## Publisher interface
Add a shared publisher interface under a repo-owned telemetry module.

Required behavior:
- safe to call from existing session code
- non-blocking in normal use
- default no-op implementation
- tolerant of partial/unavailable fields

Recommended shape:
- `publish_session_started(...)`
- `publish_load_report(...)`
- `publish_turn_finished(...)`
- `publish_runtime_sample(...)`
- `publish_log_record(...)`
- `publish_error(...)`
- `publish_session_finished(...)`

Implementation note:
- the concrete naming can differ, but the semantic event families must remain stable

## Initial data sources
### Reuse existing contracts
- token accounting from `0016`
- knob truth from `0018`
- log ring buffer from `0017`

### Backend-specific collection
Backends should populate what they can without lying.

Examples:
- HF:
  - torch allocator metrics
  - generation timing
  - model/load metadata when available
- GGUF:
  - generation timing
  - runtime/config metadata
  - token counts via llama tokenization when available
- EXL2:
  - generation timing
  - engine token IDs/usage when available
- managed vLLM:
  - server/native usage where available
  - queue and batch metrics where available

Unknown fields must remain absent rather than fabricated.
Consumers may render them as unavailable.
Requested or configured intent must not be exported as runtime truth in canonical fields.
If exported at all, requested/config values must be clearly namespaced outside canonical runtime fields.

## Export surfaces
V1 should support two export surfaces.

### 1. JSON event/export surface
Purpose:
- richer downstream dashboard consumption
- sessions, turns, logs, and inspect semantics

MVP path:
- in-process publisher
- optional file-backed JSONL sink

This is the only app export path required in v1.
Direct HTTP/SSE app export from `model-runner` is explicitly deferred.
The payload shape must remain stable regardless of sink choice.
For JSONL sinks, each line must contain exactly one complete event envelope encoded as a single-line JSON object.
JSONL output is append-only and must not rely on multiline framing.

Minimum consumer expectations:
- events are append-only
- events can be processed in timestamp order
- unknown event types are safely ignorable
- unknown payload fields are safely ignorable

### 2. Prometheus-compatible metrics export
Purpose:
- scrape-friendly numeric metrics
- Grafana compatibility
- quick validation and charting

Requirements:
- supported and disabled by default
- must not bind listeners or expose scrape surfaces unless explicitly enabled
- derived from canonical telemetry, not a separate source of truth
- limited to numeric time-series-friendly values

High-cardinality identifiers such as `event_id` and `session_id` are not part of the default Prometheus label surface.

Do not attempt to force logs, knobs, or inspect payloads into Prometheus labels.

## Metric naming guidance
Use stable, explicit names.

Recommended prefixes:
- `model_runner_session_*`
- `model_runner_turn_*`
- `model_runner_gpu_*`
- `model_runner_kv_cache_*`
- `model_runner_requests_*`
- `model_runner_tokens_*`

Labels should be bounded and practical.

Allowed examples:
- `backend`
- `engine`
- `model`
- `session_id` only if cardinality remains acceptable for local usage

Avoid:
- unbounded prompt text
- raw payload fragments
- per-message text content

## Sampling policy
Runtime sampling should be bounded.

V1 guidance:
- sample every 1s by default while a session is active
- allow configuration
- avoid heavy collectors on hot generation paths

Turn summaries should be emitted once on completion.

## Versioning
The telemetry schema must carry a version.

Requirements:
- JSON payloads include `schema_version`
- additive evolution preferred
- breaking changes require a new documented version

## Example JSON events
### `session_started`
```json
{
  "schema_version": "v1",
  "event_type": "session_started",
  "event_id": "evt_01",
  "ts": "2026-03-15T18:30:00.000Z",
  "session_id": "sess_01",
  "experiment_id": "exp_01",
  "backend_name": "hf",
  "resolved_model_id": "Qwen3.5-9B",
  "payload": {
    "session_id": "sess_01",
    "started_at": "2026-03-15T18:30:00.000Z",
    "ended_at": null,
    "status": "running",
    "activity_state": "idle",
    "backend_name": "hf",
    "transport_name": "inproc",
    "resolved_model_id": "Qwen3.5-9B",
    "config_path": "models/Qwen3.5-9B/hf/config/default.toml",
    "profile_name": "default"
  }
}
```

### `turn_finished`
```json
{
  "schema_version": "v1",
  "event_type": "turn_finished",
  "event_id": "evt_02",
  "ts": "2026-03-15T18:31:10.000Z",
  "session_id": "sess_01",
  "payload": {
    "session_id": "sess_01",
    "turn_id": 4,
    "started_at": "2026-03-15T18:31:00.000Z",
    "ended_at": "2026-03-15T18:31:10.000Z",
    "streaming_enabled": true,
    "prompt_tokens": 812,
    "completion_tokens": 146,
    "total_tokens": 958,
    "time_to_first_token_seconds": 0.72,
    "decode_latency_seconds": 9.10,
    "request_latency_seconds": 10.00,
    "stop_reason": "stop",
    "knobs": {
      "sent": {
        "temperature": 0.7,
        "top_p": 0.95
      },
      "deferred": [
        "top_k"
      ],
      "ignored": []
    }
  }
}
```

### `runtime_sample`
```json
{
  "schema_version": "v1",
  "event_type": "runtime_sample",
  "event_id": "evt_03",
  "ts": "2026-03-15T18:31:05.000Z",
  "session_id": "sess_01",
  "payload": {
    "session_id": "sess_01",
    "ts": "2026-03-15T18:31:05.000Z",
    "activity_state": "generating",
    "gpu": {
      "vram_total_bytes": 25769803776,
      "vram_used_bytes": 17179869184,
      "utilization_percent": 91.0
    },
    "throughput": {
      "tokens_generated_total": 88,
      "tokens_generated_per_second": 14.2,
      "measurement_state": "active_generation",
      "trust_level": "provisional"
    },
    "kv_cache": {
      "total_bytes": 8589934592,
      "used_bytes": 2147483648,
      "utilization_ratio": 0.25,
      "measurement_state": "available"
    },
    "requests": {
      "requests_completed_total": 3,
      "requests_in_flight": 1,
      "activity_state": "in_flight"
    }
  }
}
```

## Rollout plan
### Phase 1
- add telemetry schema module
- add publisher interface and no-op publisher
- emit session start/finish
- emit load reports
- emit completed-turn summaries

### Phase 2
- emit log events from the shared log buffer path
- add runtime sampling for supported backends
- include GPU/allocator/KV/throughput metrics where available

### Phase 3
- add optional JSON export surface
- add optional Prometheus-compatible metrics export

Implementation target for Phase 3:
- JSON export means optional JSONL sink
- Prometheus export remains disabled by default
- HTTP/SSE export is deferred

### Phase 4
- add tests for backend adapter mapping and export stability

## Testing checklist
- A runner can execute with no telemetry configured and behave normally.
- A runner with telemetry enabled emits `session_started` and `session_finished`.
- A completed turn emits canonical token/timing/knob truth when available.
- Unsupported metrics remain absent rather than fabricated.
- Prometheus export only includes numeric fields.
- Log export follows redaction rules from `0017`.

## Acceptance criteria
- This repo has a stable telemetry publisher usable by existing runners/backends.
- At least HF, GGUF, EXL2, and managed vLLM can emit baseline session and turn telemetry.
- Canonical telemetry can be consumed externally without importing runner internals.
- Prometheus export is optional, disabled by default, and derived from the canonical telemetry model.
