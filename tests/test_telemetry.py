from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    from tui_app.backends.hf import _infer_hf_finish_reason
except Exception:
    _infer_hf_finish_reason = None

from tui_app.events import TurnRecord
from tui_app.log_file import FileLogger
from tui_app.telemetry import (
    JsonlTelemetryPublisher,
    NoOpTelemetryPublisher,
    TelemetryContext,
    build_load_report_payload,
    build_runtime_sample_payload,
    resolve_model_identity,
)


class TelemetryTests(unittest.TestCase):
    def test_jsonl_publisher_writes_single_line_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "telemetry.jsonl"
            publisher = JsonlTelemetryPublisher(str(path))
            ctx = TelemetryContext(
                session_id="sess_test",
                started_at="2026-03-15T00:00:00.000Z",
                backend_name="hf",
                resolved_model_id="Qwen3.5-9B",
                resolved_model_id_kind="model_id",
                model_display_name="Qwen3.5-9B",
                model_display_name_source="resolved_model_id",
                model_path=None,
                transport_name="inproc",
                config_path=None,
                profile_name="default",
                publisher=publisher,
            )
            ctx.publish_log_record(source="backend", message="hello")

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            event = json.loads(lines[0])
            self.assertEqual(event["event_type"], "log_recorded")
            self.assertEqual(event["session_id"], "sess_test")
            self.assertEqual(event["payload"]["message"], "hello")

    def test_session_started_emits_model_display_name_and_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "telemetry.jsonl"
            args = SimpleNamespace(
                backend="hf",
                model_id="/home/poop/ml/models/Qwen3.5-9B",
                display_name="",
                _config_path="/home/poop/ml/model-runner/models/Qwen3.5-9B/hf/config/default.toml",
                _config_profile="",
                telemetry_jsonl=str(path),
            )
            session = SimpleNamespace(backend_name="hf", resolved_model_id="/home/poop/ml/models/Qwen3.5-9B")
            ctx = TelemetryContext.create(args, session)

            ctx.publish_session_started()

            event = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(event["model_display_name"], "Qwen3.5-9B")
            self.assertEqual(event["model_path"], "/home/poop/ml/models/Qwen3.5-9B")
            self.assertEqual(event["resolved_model_id_kind"], "path")
            self.assertEqual(event["payload"]["model_display_name_source"], "config_model_dir")

    def test_resolve_model_identity_falls_back_to_config_model_dir(self):
        args = SimpleNamespace(
            model_id="/home/poop/ml/models/Qwen3.5-9B",
            display_name="",
            _config_path="/home/poop/ml/model-runner/models/Qwen3.5-9B/hf/config/default.toml",
        )
        session = SimpleNamespace(resolved_model_id="/home/poop/ml/models/Qwen3.5-9B")

        identity = resolve_model_identity(args=args, session=session)

        self.assertEqual(identity.model_display_name, "Qwen3.5-9B")
        self.assertEqual(identity.model_display_name_source, "config_model_dir")
        self.assertEqual(identity.model_path, "/home/poop/ml/models/Qwen3.5-9B")
        self.assertEqual(identity.resolved_model_id_kind, "path")

    def test_load_report_carries_standardized_model_identity(self):
        args = SimpleNamespace(
            backend="hf",
            model_id="/home/poop/ml/models/Qwen3.5-9B",
            display_name="",
            _config_path="/home/poop/ml/model-runner/models/Qwen3.5-9B/hf/config/default.toml",
            max_context_tokens=49152,
            vllm_tensor_parallel_size=0,
            dtype="bfloat16",
            use_8bit=False,
            use_4bit=False,
            hf_attn_implementation="flash_attention_2",
        )
        session = SimpleNamespace(
            backend_name="hf",
            resolved_model_id="/home/poop/ml/models/Qwen3.5-9B",
            describe=lambda: {
                "torch_dtype_effective": "bfloat16",
                "weights_quantization": "none",
                "attention_backend_effective": "flash_attention_2",
                "context_length_effective": 32768,
                "memory_footprint_bytes": 1234,
                "runtime_device": "cuda:0",
                "fully_on_single_gpu": True,
                "modules_on_cpu": 0,
                "modules_on_disk": 0,
                "text_only_mode": True,
            },
        )

        payload = build_load_report_payload(session=session, args=args)

        self.assertEqual(payload["model_identifier"], "/home/poop/ml/models/Qwen3.5-9B")
        self.assertEqual(payload["model_identifier_kind"], "path")
        self.assertEqual(payload["model_display_name"], "Qwen3.5-9B")
        self.assertEqual(payload["model_path"], "/home/poop/ml/models/Qwen3.5-9B")
        self.assertEqual(payload["context_length"], 32768)
        self.assertEqual(payload["extension"]["runtime_truth"]["requested"]["context_length"], 49152)
        self.assertEqual(payload["extension"]["runtime_truth"]["confirmed"]["context_length"], 32768)
        self.assertEqual(
            payload["extension"]["runtime_truth"]["mismatches"]["context_length"],
            {"requested": 49152, "confirmed": 32768},
        )

    def test_runtime_sample_marks_idle_throughput_unavailable(self):
        ctx = TelemetryContext(
            session_id="sess_test",
            started_at="2026-03-15T00:00:00.000Z",
            backend_name="hf",
            resolved_model_id="/home/poop/ml/models/Qwen3.5-9B",
            resolved_model_id_kind="path",
            model_display_name="Qwen3.5-9B",
            model_display_name_source="config_model_dir",
            model_path="/home/poop/ml/models/Qwen3.5-9B",
            transport_name="inproc",
            config_path=None,
            profile_name="default",
            publisher=NoOpTelemetryPublisher(),
        )
        session = SimpleNamespace(describe=lambda: {})

        payload = build_runtime_sample_payload(
            telemetry=ctx,
            session=session,
            generated_tokens=42,
            elapsed_s=4.2,
            requests_completed_total=1,
            requests_in_flight=0,
        )

        self.assertEqual(payload["activity_state"], "idle")
        self.assertEqual(payload["requests"]["activity_state"], "idle")
        self.assertEqual(payload["gpu"]["measurement_state"], "unavailable")
        self.assertEqual(payload["gpu"]["diagnosis_state"], "unavailable")
        self.assertEqual(payload["throughput"]["measurement_state"], "idle")
        self.assertEqual(payload["throughput"]["reported_speed_type"], "raw_completion")
        self.assertEqual(payload["throughput"]["trust_level"], "unavailable")
        self.assertEqual(payload["throughput"]["effective_measurement_state"], "unavailable")
        self.assertEqual(payload["kv_cache"]["measurement_state"], "unavailable")
        self.assertEqual(
            payload["kv_cache"]["unavailable_reason"],
            "backend_does_not_report_kv_cache_runtime_usage",
        )
        self.assertNotIn("tokens_generated_per_second", payload["throughput"])
        self.assertNotIn("effective_tokens_per_second", payload["throughput"])

    def test_runtime_sample_marks_active_throughput_provisional(self):
        ctx = TelemetryContext(
            session_id="sess_test",
            started_at="2026-03-15T00:00:00.000Z",
            backend_name="hf",
            resolved_model_id="/home/poop/ml/models/Qwen3.5-9B",
            resolved_model_id_kind="path",
            model_display_name="Qwen3.5-9B",
            model_display_name_source="config_model_dir",
            model_path="/home/poop/ml/models/Qwen3.5-9B",
            transport_name="inproc",
            config_path=None,
            profile_name="default",
            publisher=NoOpTelemetryPublisher(),
        )
        session = SimpleNamespace(describe=lambda: {})

        payload = build_runtime_sample_payload(
            telemetry=ctx,
            session=session,
            generated_tokens=12,
            elapsed_s=3.0,
            requests_completed_total=0,
            requests_in_flight=1,
        )

        self.assertEqual(payload["activity_state"], "generating")
        self.assertEqual(payload["requests"]["activity_state"], "in_flight")
        self.assertEqual(payload["throughput"]["measurement_state"], "active_generation")
        self.assertEqual(payload["throughput"]["reported_speed_type"], "raw_completion")
        self.assertEqual(payload["throughput"]["trust_level"], "provisional")
        self.assertEqual(payload["throughput"]["effective_measurement_state"], "unavailable")
        self.assertEqual(payload["throughput"]["tokens_generated_total"], 12)
        self.assertEqual(payload["throughput"]["tokens_generated_per_second"], 4.0)
        self.assertNotIn("effective_tokens_per_second", payload["throughput"])

    def test_file_logger_subscribers_receive_canonical_message(self):
        logger = FileLogger(path=None, default_source="backend")
        seen: list[tuple[str, str]] = []
        logger.subscribe(lambda source, message: seen.append((source, message)))

        logger.log("hello\nworld", source="backend")

        self.assertEqual(seen, [("backend", "hello\\nworld")])
        self.assertRegex(logger.get_recent_logs(1)[0], r"^\d{4}-\d{2}-\d{2}T.* \[backend\] hello\\nworld$")

    def test_turn_finished_emits_ttft_and_decode_latency_when_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "telemetry.jsonl"
            ctx = TelemetryContext(
                session_id="sess_test",
                started_at="2026-03-15T00:00:00.000Z",
                backend_name="hf",
                resolved_model_id="/home/poop/ml/models/Qwen3.5-9B",
                resolved_model_id_kind="path",
                model_display_name="Qwen3.5-9B",
                model_display_name_source="config_model_dir",
                model_path="/home/poop/ml/models/Qwen3.5-9B",
                transport_name="inproc",
                config_path=None,
                profile_name="default",
                publisher=JsonlTelemetryPublisher(str(path)),
            )
            record = TurnRecord(
                raw="raw",
                think="",
                answer="answer",
                ended_in_think=False,
                backend="hf",
                model_id="/home/poop/ml/models/Qwen3.5-9B",
                gen={"finish_reason": "stop", "finish_reason_source": "local_eos_token"},
                timing={
                    "start": 1.0,
                    "end": 11.0,
                    "elapsed": 10.0,
                    "time_to_first_token": 0.75,
                },
                token_counts={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                throughput={"tokens_per_s": 0.5},
            )

            ctx.publish_turn_finished(turn_id=1, record=record)

            event = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            payload = event["payload"]
            self.assertEqual(payload["request_latency_seconds"], 10.0)
            self.assertEqual(payload["time_to_first_token_seconds"], 0.75)
            self.assertEqual(payload["decode_latency_seconds"], 9.25)
            self.assertEqual(payload["stop_reason"], "stop")
            self.assertEqual(payload["stop_reason_source"], "local_eos_token")
            self.assertEqual(payload["extension"]["throughput"]["legacy_tokens_per_second"], 0.5)
            self.assertAlmostEqual(payload["extension"]["throughput"]["raw_completion_tokens_per_second"], 5 / 9.25)
            self.assertEqual(payload["extension"]["throughput"]["effective_completion_tokens_per_second"], 0.5)

    def test_session_events_include_activity_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "telemetry.jsonl"
            ctx = TelemetryContext(
                session_id="sess_test",
                started_at="2026-03-15T00:00:00.000Z",
                backend_name="hf",
                resolved_model_id="/home/poop/ml/models/Qwen3.5-9B",
                resolved_model_id_kind="path",
                model_display_name="Qwen3.5-9B",
                model_display_name_source="config_model_dir",
                model_path="/home/poop/ml/models/Qwen3.5-9B",
                transport_name="inproc",
                config_path=None,
                profile_name="default",
                publisher=JsonlTelemetryPublisher(str(path)),
            )

            ctx.publish_session_started()
            ctx.publish_session_finished(status="finished")

            started_event, finished_event = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(started_event["payload"]["activity_state"], "idle")
            self.assertEqual(finished_event["payload"]["activity_state"], "finished")

    @unittest.skipIf(_infer_hf_finish_reason is None, "HF backend dependencies unavailable")
    def test_hf_finish_reason_infers_length_cap(self):
        finish_reason, finish_reason_source = _infer_hf_finish_reason(
            generated_token_ids=[1, 2, 3, 4],
            completion_tokens=4,
            max_new_tokens=4,
            eos_token_id=99,
        )

        self.assertEqual(finish_reason, "length")
        self.assertEqual(finish_reason_source, "local_max_new_tokens_cap")


if __name__ == "__main__":
    unittest.main()
