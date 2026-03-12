from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request

from tui_app.backends.base import EventEmitter
from tui_app.context_policy import build_context_limit_error, build_retry_report, drop_oldest_history_message, is_context_overflow_text
from tui_app.events import AnswerDelta, Error, Finish, Meta, ThinkDelta, TurnRecord, TurnStart
from tui_app.history_control import append_assistant_history
from tui_app.knobs import SUPPORTED_KNOBS, finalize_knob_report, unsupported_user_set
from tui_app.log_file import FileLogger
from tui_app.think_router import ThinkRouter

MAX_CAPTURE_BYTES = 256 * 1024


def _sanitize_for_capture(obj):
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            key_l = str(key).lower()
            if key_l in {"authorization", "api_key"}:
                out[key] = "***"
                continue
            out[key] = _sanitize_for_capture(value)
        return out
    if isinstance(obj, list):
        return [_sanitize_for_capture(x) for x in obj]
    if isinstance(obj, str) and obj.startswith("data:"):
        return "(data-url omitted)"
    return obj


def _truncate_capture(data: dict) -> dict:
    raw = json.dumps(data, ensure_ascii=False, default=str)
    if len(raw.encode("utf-8")) <= MAX_CAPTURE_BYTES:
        return data
    marker = "...(truncated)"
    keep_bytes = max(1024, MAX_CAPTURE_BYTES - len(marker.encode("utf-8")) - 64)
    trimmed = raw.encode("utf-8")[:keep_bytes].decode("utf-8", errors="ignore")
    return {"truncated": True, "max_bytes": MAX_CAPTURE_BYTES, "data": trimmed + marker}


def can_reach_ollama_host(host: str, timeout: int) -> bool:
    url = host.rstrip("/") + "/api/tags"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def detect_wsl_gateway_ip() -> str:
    try:
        out = subprocess.check_output(["ip", "route"], text=True, timeout=2)
    except Exception:
        return ""
    for line in out.splitlines():
        if line.startswith("default "):
            parts = line.split()
            if "via" in parts:
                idx = parts.index("via")
                if idx + 1 < len(parts):
                    return parts[idx + 1].strip()
    return ""


def resolve_host(cli_host: str | None, timeout: int) -> str:
    if cli_host:
        return cli_host.rstrip("/")

    candidates = []
    env_host = os.environ.get("OLLAMA_HOST", "").strip()
    if env_host:
        candidates.append(env_host)

    candidates.extend(["http://127.0.0.1:11434", "http://localhost:11434"])

    gateway_ip = detect_wsl_gateway_ip()
    if gateway_ip:
        candidates.append(f"http://{gateway_ip}:11434")

    seen = set()
    for c in candidates:
        host = c.rstrip("/")
        if host in seen:
            continue
        seen.add(host)
        if can_reach_ollama_host(host, timeout=min(timeout, 2)):
            return host

    return "http://127.0.0.1:11434"


class OllamaSession:
    backend_name = "ollama"

    def __init__(
        self,
        args: argparse.Namespace,
        resolved_model_id: str,
        host: str,
        template_info: dict[str, object] | None = None,
        logger: FileLogger | None = None,
    ):
        self.args = args
        self.resolved_model_id = resolved_model_id
        self.host = host
        self.template_info = dict(template_info or {})
        self.logger = logger
        self._last_request: dict | None = None

    def close(self) -> None:
        if self.logger is not None:
            self.logger.close()

    def get_recent_logs(self, n: int = 80, sources: list[str] | None = None) -> list[str]:
        if self.logger is None:
            return []
        return self.logger.get_recent_logs(n=n, sources=sources)

    def list_log_sources(self) -> list[str]:
        if self.logger is None:
            return []
        return self.logger.list_log_sources()

    def get_last_request(self) -> dict | None:
        return self._last_request

    def describe(self) -> dict[str, object]:
        return {
            "host": self.host,
            "model_id": self.resolved_model_id,
            "template_control_level": "server_owned_template",
            **self.template_info,
        }

    def generate_turn(self, turn_id: int, messages: list[dict[str, str]], emit: EventEmitter) -> None:
        emit(TurnStart(turn_id=turn_id))
        started = time.time()
        if self.logger is not None:
            self.logger.log(
                f"turn_start id={turn_id} messages={len(messages)} model={self.resolved_model_id} "
                f"max_new_tokens={self.args.max_new_tokens}",
                source="backend",
            )

        cli_overrides = set(getattr(self.args, "_cli_overrides", set()) or set())
        config_keys = set(getattr(self.args, "_config_keys", set()) or set())

        def _is_user_set(name: str) -> bool:
            return name in cli_overrides or name in config_keys

        options: dict[str, object] = {}
        if _is_user_set("temperature") and self.args.temperature is not None:
            options["temperature"] = self.args.temperature
        if _is_user_set("top_p") and self.args.top_p is not None:
            options["top_p"] = self.args.top_p
        if _is_user_set("top_k") and self.args.top_k is not None:
            options["top_k"] = self.args.top_k
        if _is_user_set("max_new_tokens") and self.args.max_new_tokens is not None:
            options["num_predict"] = self.args.max_new_tokens
        if _is_user_set("stop_strings") and self.args.stop_strings is not None:
            options["stop"] = self.args.stop_strings

        knob_sent: dict[str, object] = {}
        if "temperature" in options:
            knob_sent["temperature"] = options["temperature"]
        if "top_p" in options:
            knob_sent["top_p"] = options["top_p"]
        if "top_k" in options:
            knob_sent["top_k"] = options["top_k"]
        if "num_predict" in options:
            knob_sent["max_new_tokens"] = options["num_predict"]
        if "stop" in options:
            knob_sent["stop_strings"] = options["stop"]
        knob_report = finalize_knob_report(
            sent=knob_sent,
            supported=SUPPORTED_KNOBS["ollama"],
            ignored=unsupported_user_set(self.args, "ollama"),
        )

        router = ThinkRouter(assume_think=self.args.assume_think)
        working_messages = list(messages)
        overflow_retries = 0
        dropped_roles: list[str] = []
        raw_parts: list[str] = []
        think_parts: list[str] = []
        answer_parts: list[str] = []

        def _emit_generated_text(text: str) -> None:
            if not text:
                return
            token_inc = len(re.findall(r"\S+", text))
            if token_inc > 0:
                emit(Meta(turn_id=turn_id, key="generated_tokens_inc", value=token_inc))

        def _emit_think_text(text: str) -> None:
            if not text:
                return
            think_parts.append(text)
            emit(ThinkDelta(turn_id=turn_id, text=text))

        def _drop_oldest_turn() -> bool:
            dropped_chunk = drop_oldest_history_message(working_messages)
            if dropped_chunk is None:
                return False
            dropped_roles.extend(dropped_chunk)
            return True

        def _stream_chat(request_payload: dict):
            req = urllib.request.Request(
                self.host.rstrip("/") + "/api/chat",
                data=json.dumps(request_payload).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.args.ollama_timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    yield json.loads(line)

        def _consume_stream(request_payload: dict) -> None:
            for obj in _stream_chat(request_payload):
                msg = obj.get("message", {}) or {}

                thinking = msg.get("thinking", "")
                content = msg.get("content", "")
                if not isinstance(thinking, str):
                    thinking = str(thinking)
                if not isinstance(content, str):
                    content = str(content)

                if thinking:
                    _emit_generated_text(thinking)
                    raw_parts.append(thinking)
                    _emit_think_text(thinking)

                if content:
                    _emit_generated_text(content)
                    raw_parts.append(content)
                    if thinking:
                        answer_parts.append(content)
                        emit(AnswerDelta(turn_id=turn_id, text=content))
                    else:
                        for channel, text in router.feed(content):
                            if channel == "think":
                                _emit_think_text(text)
                            else:
                                answer_parts.append(text)
                                emit(AnswerDelta(turn_id=turn_id, text=text))

                if obj.get("done"):
                    break

        while True:
            payload = {
                "model": self.resolved_model_id,
                "messages": working_messages,
                "stream": True,
            }
            if options:
                payload["options"] = options
            if self.args.ollama_think in {"true", "false"}:
                payload["think"] = self.args.ollama_think == "true"
            if getattr(self.args, "capture_last_request", False):
                self._last_request = _truncate_capture(
                    _sanitize_for_capture({"backend": "ollama", "url": self.host.rstrip("/") + "/api/chat", "payload": payload})
                )
            try:
                try:
                    _consume_stream(payload)
                    break
                except urllib.error.HTTPError as exc:
                    body = exc.read().decode("utf-8", errors="ignore").strip()
                    if exc.code == 400 and "think" in payload:
                        retry_payload = dict(payload)
                        retry_payload.pop("think", None)
                        try:
                            _consume_stream(retry_payload)
                            break
                        except urllib.error.HTTPError as retry_exc:
                            retry_body = retry_exc.read().decode("utf-8", errors="ignore").strip()
                            detail = f"Ollama API request failed: HTTP {retry_exc.code} {retry_exc.reason}"
                            if retry_body:
                                detail += f" | body: {retry_body}"
                            detail += " | retry_without_think=true"
                            if not raw_parts and not think_parts and not answer_parts and is_context_overflow_text(detail):
                                if _drop_oldest_turn():
                                    overflow_retries += 1
                                    continue
                                detail = build_context_limit_error(
                                    build_retry_report(
                                        messages,
                                        working_messages,
                                        strategy="overflow_retry",
                                        context_window=None,
                                        reserved_generation_tokens=None,
                                        overflow_retries=overflow_retries,
                                        fit=False,
                                        dropped_roles=dropped_roles,
                                    )
                                )
                            if self.logger is not None:
                                self.logger.log(f"turn_error id={turn_id} {detail}", source="backend")
                            emit(Error(turn_id=turn_id, message=detail))
                            return
                    detail = f"Ollama API request failed: HTTP {exc.code} {exc.reason}"
                    if body:
                        detail += f" | body: {body}"
                    if not raw_parts and not think_parts and not answer_parts and is_context_overflow_text(detail):
                        if _drop_oldest_turn():
                            overflow_retries += 1
                            continue
                        detail = build_context_limit_error(
                            build_retry_report(
                                messages,
                                working_messages,
                                strategy="overflow_retry",
                                context_window=None,
                                reserved_generation_tokens=None,
                                overflow_retries=overflow_retries,
                                fit=False,
                                dropped_roles=dropped_roles,
                            )
                        )
                    if self.logger is not None:
                        self.logger.log(f"turn_error id={turn_id} {detail}", source="backend")
                    emit(Error(turn_id=turn_id, message=detail))
                    return
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="ignore").strip()
                detail = f"Ollama API request failed: HTTP {exc.code} {exc.reason}"
                if body:
                    detail += f" | body: {body}"
                if self.logger is not None:
                    self.logger.log(f"turn_error id={turn_id} {detail}", source="backend")
                emit(Error(turn_id=turn_id, message=detail))
                return
            except urllib.error.URLError as exc:
                if self.logger is not None:
                    self.logger.log(f"turn_error id={turn_id} Ollama API request failed: {exc}", source="backend")
                emit(Error(turn_id=turn_id, message=f"Ollama API request failed: {exc}"))
                return
            except Exception as exc:
                if not raw_parts and not think_parts and not answer_parts and is_context_overflow_text(str(exc)):
                    if _drop_oldest_turn():
                        overflow_retries += 1
                        continue
                    detail = build_context_limit_error(
                        build_retry_report(
                            messages,
                            working_messages,
                            strategy="overflow_retry",
                            context_window=None,
                            reserved_generation_tokens=None,
                            overflow_retries=overflow_retries,
                            fit=False,
                            dropped_roles=dropped_roles,
                        )
                    )
                    if self.logger is not None:
                        self.logger.log(f"turn_error id={turn_id} {detail}", source="backend")
                    emit(Error(turn_id=turn_id, message=detail))
                    return
                if self.logger is not None:
                    self.logger.log(f"turn_error id={turn_id} {exc}", source="backend")
                emit(Error(turn_id=turn_id, message=str(exc)))
                return

        for channel, text in router.flush():
            if channel == "think":
                _emit_think_text(text)
            else:
                answer_parts.append(text)
                emit(AnswerDelta(turn_id=turn_id, text=text))

        ended = time.time()
        answer_text = "".join(answer_parts)
        record = TurnRecord(
            raw="".join(raw_parts),
            think="".join(think_parts),
            answer=answer_text,
            ended_in_think=(router.mode == "think"),
            backend=self.backend_name,
            model_id=self.resolved_model_id,
            gen={
                "max_new_tokens": self.args.max_new_tokens,
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
                "top_k": self.args.top_k,
                "ollama_think": self.args.ollama_think,
            },
            timing={"start": started, "end": ended, "elapsed": max(0.0, ended - started)},
            knobs=knob_report,
            context=build_retry_report(
                messages,
                working_messages,
                strategy="overflow_retry",
                context_window=None,
                reserved_generation_tokens=None,
                overflow_retries=overflow_retries,
                fit=True,
                dropped_roles=dropped_roles,
            ).to_dict(),
            trimmed_messages=append_assistant_history(
                list(working_messages),
                think="".join(think_parts),
                answer=answer_text,
                strip_think=bool(self.args.history_strip_think),
            ),
        )
        emit(Finish(turn_id=turn_id, record=record))
        if self.logger is not None:
            self.logger.log(
                f"turn_finish id={turn_id} elapsed_s={record.timing.get('elapsed', 0):.3f} "
                f"think_chars={len(record.think)} answer_chars={len(record.answer)}",
                source="backend",
            )


def create_session(args: argparse.Namespace) -> OllamaSession:
    model_name = args.model_id
    if model_name.startswith("ollama:"):
        model_name = model_name.split(":", 1)[1]
    host = resolve_host(args.ollama_host, args.ollama_timeout)
    print(f"Using Ollama host: {host}")
    logger = FileLogger.from_value(
        getattr(args, "ollama_log_file", ""),
        "backend",
        config_path=getattr(args, "_config_path", None),
    )
    if logger is not None:
        logger.log(f"session_init model={model_name} host={host}", source="app")
    requested_template = (args.chat_template or "").strip()
    return OllamaSession(
        args=args,
        resolved_model_id=model_name,
        host=host,
        template_info={
            "chat_template_requested": requested_template,
            "chat_template_applied": False,
            "chat_template_reason": "ignored_server_owned" if requested_template else "empty_default",
        },
        logger=logger,
    )
