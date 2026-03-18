from __future__ import annotations

import json
import os
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from tui_app.events import TurnRecord


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _resolve_path(path: str, *, config_path: str | None = None) -> str:
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    if config_path:
        cfg_dir = os.path.dirname(config_path)
        if cfg_dir:
            return os.path.abspath(os.path.join(cfg_dir, expanded))
    return os.path.abspath(expanded)


def _event_id() -> str:
    return f"evt_{uuid.uuid4().hex}"


def _session_id() -> str:
    return f"sess_{uuid.uuid4().hex}"


_WINDOWS_ABS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_NVML_LOCK = threading.Lock()
_NVML_MODULE: Any | None = None
_NVML_READY = False
_NVML_INIT_ATTEMPTED = False


class TelemetryPublisher:
    def publish(self, envelope: dict[str, Any]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None


class NoOpTelemetryPublisher(TelemetryPublisher):
    def publish(self, envelope: dict[str, Any]) -> None:
        del envelope


class JsonlTelemetryPublisher(TelemetryPublisher):
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def publish(self, envelope: dict[str, Any]) -> None:
        # JSONL framing: one complete event envelope per line, append-only.
        line = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"), default=str)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def build_publisher(args: Any) -> TelemetryPublisher:
    path = str(getattr(args, "telemetry_jsonl", "") or "").strip()
    if not path:
        return NoOpTelemetryPublisher()
    return JsonlTelemetryPublisher(_resolve_path(path, config_path=getattr(args, "_config_path", None)))


@dataclass
class TelemetryContext:
    session_id: str
    started_at: str
    backend_name: str
    resolved_model_id: str
    resolved_model_id_kind: str
    model_display_name: str
    model_display_name_source: str
    model_path: str | None
    transport_name: str
    config_path: str | None
    profile_name: str
    publisher: TelemetryPublisher
    experiment_id: str | None = None

    @property
    def enabled(self) -> bool:
        return not isinstance(self.publisher, NoOpTelemetryPublisher)

    @classmethod
    def create(cls, args: Any, session: Any) -> "TelemetryContext":
        backend_name = str(getattr(session, "backend_name", getattr(args, "backend", "")) or "")
        identity = resolve_model_identity(args=args, session=session)
        transport_name = _transport_name_for_backend(backend_name)
        return cls(
            session_id=_session_id(),
            started_at=_utc_now_iso(),
            backend_name=backend_name,
            resolved_model_id=identity.resolved_model_id,
            resolved_model_id_kind=identity.resolved_model_id_kind,
            model_display_name=identity.model_display_name,
            model_display_name_source=identity.model_display_name_source,
            model_path=identity.model_path,
            transport_name=transport_name,
            config_path=getattr(args, "_config_path", None),
            profile_name=str(getattr(args, "_config_profile", "") or ""),
            publisher=build_publisher(args),
        )

    def close(self) -> None:
        self.publisher.close()

    def publish_event(self, event_type: str, payload: dict[str, Any]) -> None:
        envelope = {
            "schema_version": "v1",
            "event_type": event_type,
            "event_id": _event_id(),
            "ts": _utc_now_iso(),
            "session_id": self.session_id,
            "payload": payload,
        }
        if self.experiment_id:
            envelope["experiment_id"] = self.experiment_id
        if self.backend_name:
            envelope["backend_name"] = self.backend_name
        if self.resolved_model_id:
            envelope["resolved_model_id"] = self.resolved_model_id
        if self.resolved_model_id_kind:
            envelope["resolved_model_id_kind"] = self.resolved_model_id_kind
        if self.model_display_name:
            envelope["model_display_name"] = self.model_display_name
        if self.model_display_name_source:
            envelope["model_display_name_source"] = self.model_display_name_source
        if self.model_path:
            envelope["model_path"] = self.model_path
        try:
            self.publisher.publish(envelope)
        except Exception:
            return

    def publish_session_started(self) -> None:
        payload = {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": None,
            "status": "running",
            "activity_state": "idle",
            "activity_state_reason": "session_open_no_request_in_flight",
            "backend_name": self.backend_name,
            "transport_name": self.transport_name,
            "resolved_model_id": self.resolved_model_id,
            "resolved_model_id_kind": self.resolved_model_id_kind,
            "model_display_name": self.model_display_name,
            "model_display_name_source": self.model_display_name_source,
            "model_path": self.model_path,
            "config_path": self.config_path,
            "profile_name": self.profile_name or "",
        }
        self.publish_event("session_started", payload)

    def publish_session_finished(self, *, status: str = "finished") -> None:
        activity_state = "error" if status == "error" else "finished"
        payload = {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": _utc_now_iso(),
            "status": status,
            "activity_state": activity_state,
            "activity_state_reason": f"session_terminal_status_{status or 'finished'}",
            "backend_name": self.backend_name,
            "transport_name": self.transport_name,
            "resolved_model_id": self.resolved_model_id,
            "resolved_model_id_kind": self.resolved_model_id_kind,
            "model_display_name": self.model_display_name,
            "model_display_name_source": self.model_display_name_source,
            "model_path": self.model_path,
            "config_path": self.config_path,
            "profile_name": self.profile_name or "",
        }
        self.publish_event("session_finished", payload)

    def publish_error(self, *, scope: str, message: str) -> None:
        payload = {
            "session_id": self.session_id,
            "ts": _utc_now_iso(),
            "scope": str(scope or "session"),
            "message": str(message or ""),
        }
        self.publish_event("error_reported", payload)

    def publish_log_record(self, *, source: str, message: str) -> None:
        payload = {
            "session_id": self.session_id,
            "ts": _utc_now_iso(),
            "source": str(source or ""),
            "message": str(message or ""),
        }
        self.publish_event("log_recorded", payload)

    def publish_load_report(self, session: Any, args: Any) -> None:
        payload = build_load_report_payload(session=session, args=args)
        if payload:
            self.publish_event("load_reported", payload)

    def publish_turn_finished(self, *, turn_id: int, record: TurnRecord) -> None:
        token_counts = dict(record.token_counts or {})
        timing = dict(record.timing or {})
        throughput = dict(record.throughput or {})
        completion_tokens = token_counts.get("completion_tokens")
        payload: dict[str, Any] = {
            "session_id": self.session_id,
            "turn_id": int(turn_id),
            "started_at": _as_iso(timing.get("start")),
            "ended_at": _as_iso(timing.get("end")),
            "streaming_enabled": True,
        }
        if isinstance(token_counts.get("prompt_tokens"), int):
            payload["prompt_tokens"] = int(token_counts["prompt_tokens"])
        if isinstance(token_counts.get("completion_tokens"), int):
            payload["completion_tokens"] = int(token_counts["completion_tokens"])
        if isinstance(token_counts.get("total_tokens"), int):
            payload["total_tokens"] = int(token_counts["total_tokens"])
        request_latency = None
        decode_latency = None
        if isinstance(timing.get("elapsed"), (int, float)):
            request_latency = max(0.0, float(timing["elapsed"]))
            payload["request_latency_seconds"] = request_latency
        ttft = timing.get("time_to_first_token")
        if isinstance(ttft, (int, float)):
            ttft_value = max(0.0, float(ttft))
            payload["time_to_first_token_seconds"] = ttft_value
            if request_latency is not None:
                decode_latency = max(0.0, request_latency - ttft_value)
                payload["decode_latency_seconds"] = decode_latency
        stop_reason = record.gen.get("finish_reason") if isinstance(record.gen, dict) else None
        if stop_reason not in (None, ""):
            payload["stop_reason"] = stop_reason
        stop_reason_source = record.gen.get("finish_reason_source") if isinstance(record.gen, dict) else None
        if stop_reason_source not in (None, ""):
            payload["stop_reason_source"] = stop_reason_source
        if record.knobs is not None:
            payload["knobs"] = record.knobs
        turn_throughput = _build_turn_throughput_extension(
            throughput=throughput,
            completion_tokens=completion_tokens,
            request_latency=request_latency,
            decode_latency=decode_latency,
        )
        if turn_throughput:
            payload.setdefault("extension", {})["throughput"] = turn_throughput
        self.publish_event("turn_finished", payload)

    def publish_runtime_sample(self, payload: dict[str, Any]) -> None:
        if payload:
            self.publish_event("runtime_sample", payload)


def _transport_name_for_backend(backend_name: str) -> str:
    if backend_name in {"openai", "vllm"}:
        return "openai_http"
    if backend_name == "ollama":
        return "ollama_http"
    return "inproc"


def _as_iso(value: Any) -> str | None:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    if isinstance(value, str) and value:
        return value
    return None


def build_load_report_payload(*, session: Any, args: Any) -> dict[str, Any]:
    describe = getattr(session, "describe", None)
    info = describe() if callable(describe) else {}
    if not isinstance(info, dict):
        info = {}
    backend_name = str(getattr(session, "backend_name", getattr(args, "backend", "")) or "")
    identity = resolve_model_identity(args=args, session=session)
    payload: dict[str, Any] = {
        "engine_name": backend_name,
        "model_identifier": identity.resolved_model_id,
        "model_identifier_kind": identity.resolved_model_id_kind,
        "model_display_name": identity.model_display_name,
    }
    if identity.model_path:
        payload["model_path"] = identity.model_path

    requested_runtime = _requested_load_runtime_fields(args=args, backend_name=backend_name)
    confirmed_runtime = _confirmed_load_runtime_fields(info=info)
    for field_name in (
        "model_dtype",
        "kv_cache_dtype",
        "quantization_type",
        "attention_backend",
        "tensor_parallelism",
        "context_length",
    ):
        value = confirmed_runtime.get(field_name)
        if value not in (None, "", "unknown"):
            payload[field_name] = value

    memory_footprint = info.get("memory_footprint_bytes")
    if isinstance(memory_footprint, int):
        payload["model_weight_bytes"] = int(memory_footprint)

    extension: dict[str, Any] = {}
    runtime_truth_extension = _build_runtime_truth_extension(
        requested=requested_runtime,
        confirmed=confirmed_runtime,
    )
    if runtime_truth_extension:
        extension["runtime_truth"] = runtime_truth_extension
    if backend_name == "hf":
        extension["hf"] = {
            "runtime_device": info.get("runtime_device"),
            "fully_on_single_gpu": info.get("fully_on_single_gpu"),
            "modules_on_cpu": info.get("modules_on_cpu"),
            "modules_on_disk": info.get("modules_on_disk"),
            "text_only_mode": info.get("text_only_mode"),
        }
    elif backend_name == "vllm":
        extension["vllm"] = {
            "managed_mode": info.get("managed_mode"),
            "pid": info.get("pid"),
            "enable_auto_tool_choice": info.get("enable_auto_tool_choice"),
            "tool_call_parser": info.get("tool_call_parser"),
        }
    elif backend_name == "exl2":
        extension["exl2"] = dict(info)
    elif backend_name == "gguf":
        extension["gguf"] = dict(info)
    elif backend_name == "ollama":
        extension["ollama"] = {"host": info.get("host")}
    elif backend_name == "openai":
        extension["openai"] = {"base_url": info.get("base_url")}
    if extension:
        payload["extension"] = extension
    return payload


def _requested_load_runtime_fields(*, args: Any, backend_name: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if backend_name == "hf":
        dtype = str(getattr(args, "dtype", "") or "").strip()
        if dtype:
            payload["model_dtype"] = dtype
        payload["kv_cache_dtype"] = "none"
        payload["quantization_type"] = (
            "8bit"
            if bool(getattr(args, "use_8bit", False))
            else "4bit"
            if bool(getattr(args, "use_4bit", False))
            else "none"
        )
        payload["attention_backend"] = str(getattr(args, "hf_attn_implementation", "") or "default")
        value = getattr(args, "max_context_tokens", None)
        if isinstance(value, int) and value > 0:
            payload["context_length"] = int(value)
        return payload
    if backend_name == "vllm":
        dtype = str(getattr(args, "vllm_dtype", "") or "").strip()
        if dtype:
            payload["model_dtype"] = dtype
        payload["attention_backend"] = str(getattr(args, "vllm_attention_backend", "") or "auto")
        value = int(getattr(args, "vllm_tensor_parallel_size", 0) or 0)
        if value > 0:
            payload["tensor_parallelism"] = value
        value = int(getattr(args, "vllm_max_model_len", 0) or 0)
        if value > 0:
            payload["context_length"] = value
        return payload
    if backend_name == "gguf":
        payload["quantization_type"] = "gguf"
        payload["kv_cache_dtype"] = "none"
        value = int(getattr(args, "n_ctx", 0) or 0)
        if value > 0:
            payload["context_length"] = value
        return payload
    if backend_name == "exl2":
        payload["quantization_type"] = "exl2"
        payload["kv_cache_dtype"] = str(getattr(args, "cache_type", "") or "").strip() or "fp16"
        value = int(getattr(args, "max_seq_len", 0) or 0)
        if value > 0:
            payload["context_length"] = value
        return payload
    return payload


def _confirmed_load_runtime_fields(*, info: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    model_dtype = _first_known(
        info.get("torch_dtype_effective"),
        info.get("model_dtype_effective"),
        info.get("model_dtype"),
        info.get("vllm_dtype_effective"),
    )
    if model_dtype is not None:
        payload["model_dtype"] = model_dtype

    kv_dtype = _first_known(
        info.get("kv_cache_dtype_effective"),
        info.get("kv_cache_quantization_effective"),
        info.get("kv_cache_quantization"),
    )
    if kv_dtype is not None:
        payload["kv_cache_dtype"] = kv_dtype

    quantization = _first_known(
        info.get("weights_quantization_effective"),
        info.get("weights_quantization"),
        info.get("quantization_type"),
    )
    if quantization is not None:
        payload["quantization_type"] = quantization

    attention_backend = _first_known(
        info.get("attention_backend_effective"),
        info.get("attention_backend_runtime"),
        info.get("attention_backend"),
    )
    if attention_backend is not None:
        payload["attention_backend"] = attention_backend

    tensor_parallelism = _first_positive_int(
        info.get("tensor_parallelism_effective"),
        info.get("tensor_parallel_size_effective"),
        info.get("tensor_parallelism"),
        info.get("tensor_parallel_size"),
    )
    if tensor_parallelism is not None:
        payload["tensor_parallelism"] = tensor_parallelism

    context_length = _first_positive_int(
        info.get("context_length_effective"),
        info.get("max_model_len_effective"),
        info.get("max_seq_len_effective"),
        info.get("context_window"),
    )
    if context_length is not None:
        payload["context_length"] = context_length

    return payload


def _build_runtime_truth_extension(*, requested: dict[str, Any], confirmed: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if requested:
        payload["requested"] = requested
    if confirmed:
        payload["confirmed"] = confirmed
    mismatches: dict[str, dict[str, Any]] = {}
    unconfirmed_requested: list[str] = []
    for key, requested_value in requested.items():
        if not _is_concrete_requested_value(requested_value):
            continue
        confirmed_value = confirmed.get(key)
        if confirmed_value is None:
            unconfirmed_requested.append(key)
            continue
        if confirmed_value != requested_value:
            mismatches[key] = {
                "requested": requested_value,
                "confirmed": confirmed_value,
            }
    if mismatches:
        payload["mismatches"] = mismatches
    if unconfirmed_requested:
        payload["unconfirmed_requested"] = sorted(unconfirmed_requested)
    return payload


def _is_concrete_requested_value(value: Any) -> bool:
    if value in (None, "", "auto", "default", "model_default", "server_default", "server_managed", "(unset)"):
        return False
    return True


def _first_known(*values: Any) -> Any | None:
    for value in values:
        if value not in (None, "", "unknown"):
            return value
    return None


def _first_positive_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return int(value)
        if isinstance(value, str) and value.isdigit():
            parsed = int(value)
            if parsed > 0:
                return parsed
    return None


def _build_turn_throughput_extension(
    *,
    throughput: dict[str, Any],
    completion_tokens: Any,
    request_latency: float | None,
    decode_latency: float | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    completion_token_count = int(completion_tokens) if isinstance(completion_tokens, int) and completion_tokens >= 0 else None
    legacy_tokens_per_second = throughput.get("tokens_per_s")
    if isinstance(legacy_tokens_per_second, (int, float)):
        payload["legacy_tokens_per_second"] = float(legacy_tokens_per_second)
        if completion_token_count is not None and request_latency is not None and request_latency > 0:
            payload["legacy_measurement_basis"] = "completion_tokens_div_request_latency_seconds"
    if completion_token_count is not None and decode_latency is not None and decode_latency > 0:
        payload["raw_completion_tokens_per_second"] = float(completion_token_count / decode_latency)
        payload["raw_measurement_basis"] = "completion_tokens_div_decode_latency_seconds"
    else:
        payload["raw_measurement_state"] = "unavailable"
        payload["raw_unavailable_reason"] = "decode_latency_missing"
    if completion_token_count is not None and request_latency is not None and request_latency > 0:
        payload["effective_completion_tokens_per_second"] = float(completion_token_count / request_latency)
        payload["effective_measurement_basis"] = "completion_tokens_div_request_latency_seconds"
    else:
        payload["effective_measurement_state"] = "unavailable"
        payload["effective_unavailable_reason"] = "request_latency_or_completion_tokens_missing"
    return payload


def build_runtime_sample_payload(
    *,
    telemetry: TelemetryContext,
    session: Any,
    generated_tokens: int,
    elapsed_s: float,
    requests_completed_total: int,
    requests_in_flight: int,
) -> dict[str, Any]:
    requests_in_flight_count = int(max(0, requests_in_flight))
    payload: dict[str, Any] = {
        "session_id": telemetry.session_id,
        "ts": _utc_now_iso(),
        "activity_state": _activity_state_for_requests(requests_in_flight_count),
        "activity_state_reason": _activity_state_reason_for_requests(requests_in_flight_count),
    }
    groups: dict[str, Any] = {}

    describe = getattr(session, "describe", None)
    info = describe() if callable(describe) else {}
    if not isinstance(info, dict):
        info = {}

    gpu_group: dict[str, Any] = {}
    memory_group: dict[str, Any] = {}
    runtime_device = str(info.get("runtime_device") or "")
    try:
        import torch  # type: ignore

        cuda_index = None
        if runtime_device.startswith("cuda:"):
            cuda_index = int(runtime_device.split(":", 1)[1])
        elif runtime_device == "cuda" and torch.cuda.is_available():
            cuda_index = int(torch.cuda.current_device())
        if cuda_index is not None and torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info(cuda_index)
            used_bytes = int(total_bytes - free_bytes)
            gpu_group.update(
                {
                    "vram_total_bytes": int(total_bytes),
                    "vram_used_bytes": used_bytes,
                    "vram_free_bytes": int(free_bytes),
                }
            )
            gpu_group.update(_read_nvml_gpu_metrics(cuda_index))
            try:
                memory_group["torch_cuda_memory_allocated_bytes"] = int(torch.cuda.memory_allocated(cuda_index))
                memory_group["torch_cuda_memory_reserved_bytes"] = int(torch.cuda.memory_reserved(cuda_index))
                memory_group["torch_cuda_max_allocated_bytes"] = int(torch.cuda.max_memory_allocated(cuda_index))
            except Exception:
                pass
    except Exception:
        pass

    groups["gpu"] = _finalize_gpu_group(gpu_metrics=gpu_group, runtime_device=runtime_device)
    if memory_group:
        groups["memory"] = memory_group

    groups["kv_cache"] = _build_kv_cache_group(info)
    groups["throughput"] = _build_throughput_group(
        requests_in_flight=requests_in_flight_count,
        generated_tokens=generated_tokens,
        elapsed_s=elapsed_s,
    )
    groups["requests"] = {
        "requests_completed_total": int(max(0, requests_completed_total)),
        "requests_in_flight": requests_in_flight_count,
        "activity_state": "in_flight" if requests_in_flight_count > 0 else "idle",
    }
    payload.update(groups)
    return payload


def attach_log_subscribers(session: Any, callback: Callable[[str, str], None]) -> None:
    seen: set[int] = set()

    def _attach(logger: Any) -> None:
        if logger is None:
            return
        marker = id(logger)
        if marker in seen:
            return
        seen.add(marker)
        subscribe = getattr(logger, "subscribe", None)
        if callable(subscribe):
            subscribe(callback)

    _attach(getattr(session, "logger", None))
    transport = getattr(session, "_transport", None)
    if transport is not None:
        _attach(getattr(transport, "logger", None))


@dataclass(frozen=True)
class ModelIdentity:
    resolved_model_id: str
    resolved_model_id_kind: str
    model_display_name: str
    model_display_name_source: str
    model_path: str | None = None


def resolve_model_identity(*, args: Any, session: Any) -> ModelIdentity:
    resolved_model_id = str(getattr(session, "resolved_model_id", getattr(args, "model_id", "")) or "").strip()
    model_path = _normalize_model_path(resolved_model_id) if _looks_like_local_path(resolved_model_id) else None
    display_name = ""
    display_name_source = ""

    configured_display_name = str(getattr(args, "display_name", "") or "").strip()
    if configured_display_name:
        display_name = configured_display_name
        display_name_source = "config_display_name"
    else:
        config_model_name = _model_name_from_config_path(getattr(args, "_config_path", None))
        if config_model_name:
            display_name = config_model_name
            display_name_source = "config_model_dir"
        elif resolved_model_id and not model_path:
            display_name = resolved_model_id
            display_name_source = "resolved_model_id"
        elif model_path:
            display_name = os.path.basename(model_path.rstrip("/\\")) or model_path
            display_name_source = "path_basename"

    return ModelIdentity(
        resolved_model_id=resolved_model_id,
        resolved_model_id_kind="path" if model_path else "model_id",
        model_display_name=display_name,
        model_display_name_source=display_name_source,
        model_path=model_path,
    )


def _model_name_from_config_path(config_path: str | None) -> str:
    if not config_path:
        return ""
    parts = os.path.normpath(str(config_path)).split(os.sep)
    try:
        idx = parts.index("models")
    except ValueError:
        return ""
    if idx + 1 >= len(parts):
        return ""
    return str(parts[idx + 1] or "")


def _normalize_model_path(value: str) -> str:
    expanded = os.path.expanduser(value.strip())
    if _WINDOWS_ABS_PATH.match(expanded):
        drive = expanded[0].lower()
        rest = expanded[2:].lstrip("\\/").replace("\\", "/")
        expanded = f"/mnt/{drive}/{rest}"
    if os.path.isabs(expanded) or os.path.exists(expanded):
        return os.path.abspath(expanded)
    return expanded


def _looks_like_local_path(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    expanded = os.path.expanduser(text)
    return bool(
        os.path.isabs(expanded)
        or expanded.startswith((".", "~"))
        or _WINDOWS_ABS_PATH.match(expanded)
        or expanded.lower().endswith(".gguf")
        or os.path.exists(expanded)
    )


def _build_throughput_group(*, requests_in_flight: int, generated_tokens: int, elapsed_s: float) -> dict[str, Any]:
    if int(max(0, requests_in_flight)) <= 0:
        return {
            "reported_speed_type": "raw_completion",
            "measurement_state": "idle",
            "trust_level": "unavailable",
            "trust_reason": "no_active_generation",
            "effective_measurement_state": "unavailable",
            "effective_unavailable_reason": "end_to_end_request_speed_not_available_without_active_generation",
        }

    payload: dict[str, Any] = {
        "reported_speed_type": "raw_completion",
        "measurement_state": "active_generation",
        "measurement_basis": "current_turn_completion_tokens_div_elapsed_seconds",
        "trust_level": "provisional",
        "trust_reason": "derived_from_streamed_completion_tokens_and_local_elapsed_time",
        "effective_measurement_state": "unavailable",
        "effective_unavailable_reason": "end_to_end_request_speed_not_known_until_turn_completion",
        "tokens_generated_total": int(max(0, generated_tokens)),
    }
    if elapsed_s > 0:
        payload["sample_window_seconds"] = float(elapsed_s)
        payload["tokens_generated_per_second"] = float(generated_tokens / elapsed_s)
    return payload


def _build_kv_cache_group(info: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    numeric_fields = {
        "total_bytes": info.get("kv_cache_total_bytes"),
        "used_bytes": info.get("kv_cache_used_bytes"),
        "utilization_ratio": info.get("kv_cache_utilization_ratio"),
        "block_size": info.get("kv_cache_block_size"),
        "sequence_count": info.get("kv_cache_sequence_count"),
    }
    for key, value in numeric_fields.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            payload[key] = int(value)
        elif isinstance(value, float):
            payload[key] = float(value)
    if "utilization_ratio" not in payload:
        total_bytes = payload.get("total_bytes")
        used_bytes = payload.get("used_bytes")
        if isinstance(total_bytes, int) and total_bytes > 0 and isinstance(used_bytes, int) and used_bytes >= 0:
            payload["utilization_ratio"] = float(used_bytes / total_bytes)
    if payload:
        payload["measurement_state"] = "available"
        return payload
    return {
        "measurement_state": "unavailable",
        "unavailable_reason": "backend_does_not_report_kv_cache_runtime_usage",
    }


def _activity_state_for_requests(requests_in_flight: int) -> str:
    return "generating" if int(max(0, requests_in_flight)) > 0 else "idle"


def _activity_state_reason_for_requests(requests_in_flight: int) -> str:
    if int(max(0, requests_in_flight)) > 0:
        return "requests_in_flight_positive"
    return "no_requests_in_flight"


def _finalize_gpu_group(*, gpu_metrics: dict[str, Any], runtime_device: str) -> dict[str, Any]:
    if not runtime_device or not (runtime_device == "cuda" or runtime_device.startswith("cuda:")):
        return {
            "measurement_state": "unavailable",
            "unavailable_reason": "session_not_running_on_cuda",
            "diagnosis_state": "unavailable",
            "diagnosis_reason": "gpu_not_active_for_this_session",
        }

    payload = dict(gpu_metrics)
    if not payload:
        return {
            "measurement_state": "unavailable",
            "unavailable_reason": "cuda_metrics_collection_failed",
            "diagnosis_state": "incomplete",
            "diagnosis_reason": "gpu_runtime_metrics_unavailable_for_operator_diagnosis",
        }

    has_memory_view = all(key in payload for key in ("vram_total_bytes", "vram_used_bytes", "vram_free_bytes"))
    has_utilization_view = "utilization_percent" in payload
    payload["measurement_state"] = "available" if has_memory_view and has_utilization_view else "partial"
    payload["diagnosis_state"] = "incomplete"
    payload["diagnosis_reason"] = "gpu_runtime_metrics_do_not_cover_compute_memory_queue_interpretation"
    return payload


def _read_nvml_gpu_metrics(cuda_index: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    pynvml = _load_pynvml()
    if pynvml is None:
        return metrics
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(cuda_index))
    except Exception:
        return metrics

    try:
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = getattr(utilization, "gpu", None)
        if gpu_util is not None:
            metrics["utilization_percent"] = float(gpu_util)
    except Exception:
        pass

    try:
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        metrics["power_usage_watts"] = float(power_mw) / 1000.0
    except Exception:
        pass

    try:
        temp_c = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        metrics["temperature_celsius"] = int(temp_c)
    except Exception:
        pass

    return metrics


def _load_pynvml() -> Any | None:
    global _NVML_MODULE, _NVML_READY, _NVML_INIT_ATTEMPTED
    if _NVML_READY:
        return _NVML_MODULE
    if _NVML_INIT_ATTEMPTED:
        return None
    with _NVML_LOCK:
        if _NVML_READY:
            return _NVML_MODULE
        if _NVML_INIT_ATTEMPTED:
            return None
        _NVML_INIT_ATTEMPTED = True
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            _NVML_MODULE = pynvml
            _NVML_READY = True
            return _NVML_MODULE
        except Exception:
            _NVML_MODULE = None
            _NVML_READY = False
            return None
