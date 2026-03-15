from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from config_utils import apply_machine_model_root, load_config_layers
from tui_app.backends.base import EventEmitter
from tui_app.context_policy import build_context_limit_error, reserve_generation_tokens, trim_messages_to_budget
from tui_app.events import Error, Finish, TurnStart
from tui_app.log_file import FileLogger
from tui_app.transports.openai_http import OpenAIHTTPSession, normalize_openai_base_url

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - transformers is a required runtime elsewhere in this repo
    AutoTokenizer = None  # type: ignore


def _resolve_api_key(args: argparse.Namespace) -> str:
    cli_overrides = set(getattr(args, "_cli_overrides", set()) or set())
    if "vllm_api_key" in cli_overrides:
        return (args.vllm_api_key or "").strip()
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    return (args.vllm_api_key or "").strip()


def _pick_free_port(host: str) -> int:
    bind_host = host if host not in {"0.0.0.0", "::"} else "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((bind_host, 0))
        return int(sock.getsockname()[1])


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def _json_get(url: str, *, timeout_s: float, api_key: str = "") -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, method="GET", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    import json

    return json.loads(raw)


def _resolve_model_from_models_endpoint(base_url: str, *, timeout_s: float, api_key: str, served_model_name: str) -> str:
    if served_model_name:
        return served_model_name
    data = _json_get(_join_url(base_url, "/models"), timeout_s=max(1.0, min(10.0, timeout_s)), api_key=api_key)
    models = data.get("data")
    ids: list[str] = []
    if isinstance(models, list):
        for item in models:
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip():
                ids.append(item["id"].strip())
    if len(ids) == 1:
        return ids[0]
    if not ids:
        raise RuntimeError("vLLM model resolution failed: /v1/models returned no models.")
    raise RuntimeError(
        "vLLM model resolution failed: server returned multiple models. "
        f"Set backend.vllm.served_model_name. Available: {', '.join(ids)}"
    )


def _supports_chat_template_flag(cmd_argv: list[str]) -> bool:
    probe = list(cmd_argv)
    if "serve" in probe:
        probe = probe[: probe.index("serve") + 1]
    else:
        probe.append("serve")
    for help_flag in ("--help", "--help=all"):
        try:
            out = subprocess.check_output(probe + [help_flag], text=True, stderr=subprocess.STDOUT, timeout=8)
        except Exception:
            continue
        if "--chat-template" in out:
            return True
    return False


def _slugify(value: str) -> str:
    text = "".join(ch if ch.isalnum() else "-" for ch in (value or "").strip().lower())
    text = "-".join(part for part in text.split("-") if part)
    return text or "vllm"


def _resolve_sidecar_path(args: argparse.Namespace, filename: str) -> str:
    config_path = (getattr(args, "_config_path", "") or "").strip()
    if config_path:
        base_dir = os.path.join(os.path.dirname(config_path), "run")
        return os.path.abspath(os.path.join(base_dir, filename))
    stem = _slugify((getattr(args, "vllm_served_model_name", "") or getattr(args, "model_id", "") or "vllm"))
    return os.path.abspath(os.path.join("/tmp", f"model-runner-{stem}-{filename}"))


def _resolve_engine_log_paths(args: argparse.Namespace) -> tuple[str, str]:
    config_path = (getattr(args, "_config_path", "") or "").strip()
    base_dir = ""
    if config_path:
        base_dir = os.path.join(os.path.dirname(config_path), "logs")
    else:
        stem = _slugify((getattr(args, "vllm_served_model_name", "") or getattr(args, "model_id", "") or "vllm"))
        base_dir = os.path.join("/tmp", f"model-runner-{stem}-logs")
    return (
        os.path.abspath(os.path.join(base_dir, "vllm-engine.stdout.log")),
        os.path.abspath(os.path.join(base_dir, "vllm-engine.stderr.log")),
    )


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _stop_pid_group(pid: int, pgid: int | None = None) -> None:
    if pid <= 0:
        return
    target_pgid = pgid
    if target_pgid is None:
        try:
            target_pgid = os.getpgid(pid)
        except Exception:
            target_pgid = None
    if target_pgid is not None:
        try:
            os.killpg(target_pgid, signal.SIGTERM)
        except Exception:
            pass
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if not _pid_alive(pid):
                return
            time.sleep(0.1)
        try:
            os.killpg(target_pgid, signal.SIGKILL)
        except Exception:
            pass
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:
        return


def _resolve_requested_template(args: argparse.Namespace) -> str:
    requested_template = (args.chat_template or "").strip()
    if not requested_template:
        return ""
    config_path = (getattr(args, "_config_path", "") or "").strip()
    candidate = os.path.abspath(os.path.expanduser(requested_template))
    if config_path and not os.path.isabs(os.path.expanduser(requested_template)):
        candidate = os.path.abspath(os.path.join(os.path.dirname(config_path), requested_template))
    if os.path.isfile(candidate):
        return candidate
    return requested_template


def _build_signature(args: argparse.Namespace, model_id: str, template_requested_value: str) -> dict[str, Any]:
    extra = args.vllm_extra_args or []
    if isinstance(extra, str):
        extra = shlex.split(extra)
    return {
        "backend": "vllm",
        "mode": str(args.vllm_mode or "managed"),
        "model_id": model_id,
        "host": str(args.vllm_host or "127.0.0.1"),
        "requested_port": int(args.vllm_port),
        "cmd": str(args.vllm_cmd or "vllm"),
        "served_model_name": str(args.vllm_served_model_name or ""),
        "tensor_parallel_size": int(args.vllm_tensor_parallel_size or 0),
        "gpu_memory_utilization": float(args.vllm_gpu_memory_utilization or 0),
        "max_model_len": int(args.vllm_max_model_len or 0),
        "generation_config": str(args.vllm_generation_config or ""),
        "attention_backend": str(args.vllm_attention_backend or ""),
        "dtype": str(args.vllm_dtype or ""),
        "enable_auto_tool_choice": bool(args.vllm_enable_auto_tool_choice is True),
        "tool_call_parser": str(args.vllm_tool_call_parser or ""),
        "extra_args": [str(x) for x in extra if str(x).strip()],
        "chat_template_requested": template_requested_value,
    }


def _load_control_file(path: str) -> dict[str, Any] | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _write_control_file(path: str, data: dict[str, Any]) -> None:
    _ensure_parent_dir(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _remove_control_file(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    except Exception:
        return


class _FileTail:
    def __init__(self, logger: FileLogger, source: str, path: str, start_offset: int, stop_event: threading.Event):
        self.logger = logger
        self.source = source
        self.path = path
        self.start_offset = max(0, int(start_offset))
        self.stop_event = stop_event

    def pump(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as stream:
                stream.seek(self.start_offset)
                while not self.stop_event.is_set():
                    line = stream.readline()
                    if line:
                        self.logger.log(line.rstrip(), source=self.source)
                        continue
                    time.sleep(0.15)
        except Exception:
            return


class VLLMSession:
    backend_name = "vllm"

    def __init__(
        self,
        args: argparse.Namespace,
        process: subprocess.Popen,
        launch_argv: list[str],
        base_url: str,
        resolved_model_id: str,
        api_key: str,
        template_info: dict[str, object],
        logger: FileLogger,
        control_file_path: str,
        managed_pid: int,
        managed_pgid: int | None,
        attached_existing: bool,
        engine_stdout_start: int = 0,
        engine_stderr_start: int = 0,
        tail_stop_event: threading.Event | None = None,
        tail_threads: list[threading.Thread] | None = None,
        tokenizer=None,
    ):
        self.args = args
        self._process = process
        self._managed_pid = int(managed_pid)
        self._managed_pgid = managed_pgid
        self._launch_argv = launch_argv
        self._base_url = base_url
        self.logger = logger
        self._control_file_path = control_file_path
        self._engine_stdout_path = ""
        self._engine_stderr_path = ""
        self._engine_stdout_start = max(0, int(engine_stdout_start))
        self._engine_stderr_start = max(0, int(engine_stderr_start))
        self._attached_existing = bool(attached_existing)
        self._tail_stop_event = tail_stop_event
        self._tail_threads = list(tail_threads or [])
        self._shutdown_on_close = True
        self._closed = False
        self._tokenizer = tokenizer
        self._transport = OpenAIHTTPSession(
            args=args,
            resolved_model_id=resolved_model_id,
            base_url=base_url,
            api_key=api_key,
            timeout_s=float(args.vllm_timeout_s),
            backend_name="vllm",
            template_info=template_info,
            logger=logger,
        )
        self.resolved_model_id = resolved_model_id

    def describe(self) -> dict[str, object]:
        info = dict(self._transport.describe())
        info["managed_mode"] = True
        info["pid"] = self._managed_pid
        info["launch_argv"] = " ".join(shlex.quote(part) for part in self._launch_argv)
        info["enable_auto_tool_choice"] = self.args.vllm_enable_auto_tool_choice
        info["tool_call_parser"] = self.args.vllm_tool_call_parser or ""
        info["control_file"] = self._control_file_path
        info["engine_stdout_path"] = self._engine_stdout_path
        info["engine_stderr_path"] = self._engine_stderr_path
        info["engine_stdout_start"] = self._engine_stdout_start
        info["engine_stderr_start"] = self._engine_stderr_start
        info["attached_existing"] = self._attached_existing
        return info

    def get_recent_logs(self, n: int = 80, sources: list[str] | None = None) -> list[str]:
        return self.logger.get_recent_logs(n=n, sources=sources)

    def list_log_sources(self) -> list[str]:
        return self.logger.list_log_sources()

    def get_last_request(self) -> dict | None:
        getter = getattr(self._transport, "get_last_request", None)
        if callable(getter):
            return getter()
        return None

    def detach(self) -> None:
        self._shutdown_on_close = False

    @staticmethod
    def _sanitize_messages_for_preflight(messages: list[dict[str, object]]) -> list[dict[str, object]] | None:
        clean: list[dict[str, object]] = []
        for msg in messages:
            role = str(msg.get("role", "") or "").strip()
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            images = msg.get("images")
            if isinstance(images, list) and images:
                return None
            content_obj = msg.get("content")
            clean_msg: dict[str, object] = {
                "role": role,
                "content": content_obj if content_obj is None or isinstance(content_obj, str) else str(content_obj),
            }
            tool_calls = msg.get("tool_calls")
            if role == "assistant" and isinstance(tool_calls, list):
                clean_msg["tool_calls"] = tool_calls
            tool_call_id = msg.get("tool_call_id")
            if role == "tool" and isinstance(tool_call_id, str) and tool_call_id.strip():
                clean_msg["tool_call_id"] = tool_call_id.strip()
            clean.append(clean_msg)
        return clean

    def _preflight_context(self, messages: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object] | None]:
        tokenizer = self._tokenizer
        if tokenizer is None or getattr(tokenizer, "chat_template", None) is None:
            return list(messages), None
        clean = self._sanitize_messages_for_preflight(messages)
        if clean is None:
            return list(messages), None

        context_window = int(self.args.vllm_max_model_len or 0) or None
        if context_window is None:
            value = getattr(tokenizer, "model_max_length", None)
            if isinstance(value, int) and 0 < value < 1_000_000:
                context_window = int(value)
        if context_window is None:
            return list(messages), None

        def _measure(trimmed: list[dict[str, object]]):
            token_ids = tokenizer.apply_chat_template(
                trimmed,
                tokenize=True,
                add_generation_prompt=True,
            )
            return int(len(token_ids)), None

        reserved = reserve_generation_tokens(context_window, self.args.max_new_tokens)

        trimmed, _prompt_tokens, _unused, report = trim_messages_to_budget(
            clean,
            measure_fn=_measure,
            context_window=context_window,
            reserved_generation_tokens=reserved,
            strategy="exact_preflight",
        )
        if not report.fit:
            raise RuntimeError(build_context_limit_error(report))
        return list(trimmed), report.to_dict()

    @staticmethod
    def _merge_context_reports(
        preflight: dict[str, object] | None,
        fallback: dict[str, object] | None,
        *,
        token_counts: dict[str, int] | None,
    ) -> dict[str, object] | None:
        if preflight is None:
            return fallback
        if fallback is None:
            return preflight
        pre = dict(preflight)
        post = dict(fallback)
        combined_roles = list(pre.get("dropped_roles") or []) + list(post.get("dropped_roles") or [])
        prompt_tokens = post.get("prompt_tokens")
        if prompt_tokens is None and isinstance(token_counts, dict):
            prompt_tokens = token_counts.get("prompt_tokens")
        strategy = str(pre.get("strategy") or "exact_preflight")
        post_dropped = int(post.get("dropped_messages") or 0)
        if post_dropped > 0 or int(post.get("overflow_retries") or 0) > 0:
            strategy = f"{strategy}+overflow_retry"
        return {
            "strategy": strategy,
            "original_messages": pre.get("original_messages", post.get("original_messages")),
            "kept_messages": post.get("kept_messages", pre.get("kept_messages")),
            "dropped_messages": int(pre.get("dropped_messages") or 0) + post_dropped,
            "dropped_roles": combined_roles,
            "fit": post.get("fit", pre.get("fit")),
            "context_window": pre.get("context_window", post.get("context_window")),
            "reserved_generation_tokens": pre.get("reserved_generation_tokens", post.get("reserved_generation_tokens")),
            "prompt_budget_tokens": pre.get("prompt_budget_tokens", post.get("prompt_budget_tokens")),
            "prompt_tokens": prompt_tokens if prompt_tokens is not None else pre.get("prompt_tokens"),
            "overflow_retries": post.get("overflow_retries", 0),
            "system_message_present": pre.get("system_message_present", post.get("system_message_present")),
            "system_message_preserved": bool(pre.get("system_message_preserved", True))
            and bool(post.get("system_message_preserved", True)),
            "system_drop_required": bool(pre.get("system_drop_required", False))
            or bool(post.get("system_drop_required", False)),
        }

    def generate_turn(self, turn_id: int, messages: list[dict[str, object]], emit: EventEmitter) -> None:
        process_dead = False
        if self._process is not None:
            process_dead = self._process.poll() is not None
        else:
            process_dead = not _pid_alive(self._managed_pid)
        if process_dead:
            tail_rows = self.logger.get_recent_logs(40)
            tail = "\n".join(tail_rows) if tail_rows else "(no captured logs)"
            emit(
                Error(
                    turn_id=turn_id,
                    message=(
                        "vLLM managed server exited unexpectedly before generation.\n"
                        f"launch_argv: {' '.join(shlex.quote(p) for p in self._launch_argv)}\n"
                        f"log_tail:\n{tail}"
                    ),
                )
            )
            return
        emit(TurnStart(turn_id=turn_id))
        try:
            working_messages, preflight_context = self._preflight_context(messages)
        except Exception as exc:
            emit(Error(turn_id=turn_id, message=str(exc)))
            return

        def _forward(ev):
            if isinstance(ev, TurnStart):
                return
            if isinstance(ev, Finish):
                ev.record.context = self._merge_context_reports(
                    preflight_context,
                    ev.record.context,
                    token_counts=ev.record.token_counts,
                )
            emit(ev)

        self._transport.generate_turn(
            turn_id=turn_id,
            messages=working_messages,
            emit=_forward,
            emit_turn_start=False,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._tail_stop_event is not None:
            self._tail_stop_event.set()
        for thread in self._tail_threads:
            try:
                thread.join(timeout=0.5)
            except Exception:
                pass
        if self._shutdown_on_close:
            _stop_pid_group(self._managed_pid, self._managed_pgid)
            _remove_control_file(self._control_file_path)
        self.logger.close()


def _build_launch_argv(args: argparse.Namespace, host: str, port: int, model_id: str) -> list[str]:
    cmd_raw = (args.vllm_cmd or "vllm").strip()
    cmd_argv = shlex.split(cmd_raw) if cmd_raw else []
    if not cmd_argv:
        cmd_argv = ["vllm"]
    # Robust default: if plain `vllm` is unavailable on PATH, use current interpreter.
    if cmd_argv == ["vllm"] and shutil.which("vllm") is None:
        cmd_argv = [sys.executable, "-m", "vllm.entrypoints.cli.main"]
    if not cmd_argv:
        raise RuntimeError("vLLM command is empty (backend.vllm.cmd / --vllm-cmd).")

    launch = list(cmd_argv)
    if "serve" not in launch:
        launch.append("serve")
    launch.extend([model_id, "--host", host, "--port", str(port)])

    if args.vllm_served_model_name:
        launch.extend(["--served-model-name", str(args.vllm_served_model_name)])
    if int(args.vllm_tensor_parallel_size or 0) > 0:
        launch.extend(["--tensor-parallel-size", str(args.vllm_tensor_parallel_size)])
    if float(args.vllm_gpu_memory_utilization or 0) > 0:
        launch.extend(["--gpu-memory-utilization", str(args.vllm_gpu_memory_utilization)])
    if int(args.vllm_max_model_len or 0) > 0:
        launch.extend(["--max-model-len", str(args.vllm_max_model_len)])
    generation_config = (args.vllm_generation_config or "").strip() or "auto"
    if generation_config == "hf":
        generation_config = "auto"
    if generation_config:
        launch.extend(["--generation-config", generation_config])
    if (args.vllm_attention_backend or "").strip():
        launch.extend(["--attention-backend", str(args.vllm_attention_backend).strip()])
    if (args.vllm_dtype or "").strip():
        launch.extend(["--dtype", str(args.vllm_dtype).strip()])
    if args.vllm_enable_auto_tool_choice is True:
        launch.append("--enable-auto-tool-choice")
    parser_name = str(args.vllm_tool_call_parser or "").strip()
    if parser_name:
        launch.extend(["--tool-call-parser", parser_name])

    extra = args.vllm_extra_args or []
    if isinstance(extra, str):
        extra = shlex.split(extra)
    if isinstance(extra, list):
        launch.extend([str(x) for x in extra if str(x).strip()])
    if "--enable-force-include-usage" not in launch:
        launch.append("--enable-force-include-usage")
    return launch


def _wait_until_ready(
    process: subprocess.Popen | None,
    *,
    base_url: str,
    timeout_s: float,
    api_key: str,
) -> None:
    deadline = time.time() + max(5.0, min(120.0, timeout_s))
    err_last = ""
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError("vLLM process exited before readiness probe succeeded.")
        try:
            _json_get(_join_url(base_url, "/models"), timeout_s=2.0, api_key=api_key)
            return
        except urllib.error.HTTPError as exc:
            err_last = f"HTTP {exc.code} {exc.reason}"
        except Exception as exc:
            err_last = str(exc)
        time.sleep(0.4)
    raise RuntimeError(f"Timed out waiting for readiness at {_join_url(base_url, '/models')}: {err_last}")


def create_session(args: argparse.Namespace) -> VLLMSession:
    mode = (args.vllm_mode or "managed").strip().lower()
    if mode != "managed":
        raise RuntimeError(f"Unsupported vLLM mode: {mode}. v1 supports managed mode only.")

    model_id = (args.model_id or "").strip()
    # Safety net: resolve local stems even if positional precedence bypasses
    # config-layer model_id rewriting.
    if model_id:
        rewritten = apply_machine_model_root(model_id)
        if rewritten:
            model_id = rewritten
    # Secondary fallback: reload merged config and honor model_id from it.
    cfg_path = (getattr(args, "_config_path", "") or "").strip()
    if cfg_path:
        try:
            merged, _meta = load_config_layers(
                cfg_path,
                backend="vllm",
                profile=(getattr(args, "_config_profile", "") or ""),
                include_machine=True,
            )
            cfg_model_id = (merged.get("model_id") or "").strip() if isinstance(merged, dict) else ""
            if not model_id and cfg_model_id:
                model_id = apply_machine_model_root(cfg_model_id)
        except Exception:
            pass
    if not model_id:
        raise RuntimeError("vLLM managed mode requires model_id (path or model name).")

    host = (args.vllm_host or "127.0.0.1").strip()
    requested_port = int(args.vllm_port)
    if requested_port < 0:
        raise RuntimeError("vLLM port must be >= 0.")
    api_key = _resolve_api_key(args)
    template_requested_value = _resolve_requested_template(args)
    signature = _build_signature(args, model_id, template_requested_value)
    control_file_path = _resolve_sidecar_path(args, "vllm-managed.json")
    existing = _load_control_file(control_file_path)
    if existing:
        existing_pid = int(existing.get("pid") or 0)
        if not _pid_alive(existing_pid):
            _remove_control_file(control_file_path)
            existing = None
        elif existing.get("signature") != signature:
            raise RuntimeError(
                "Detached managed vLLM is already running for this config slot with different settings. "
                f"Stop it first with: python tui.py --config {shlex.quote((getattr(args, '_config_path', '') or getattr(args, 'model_id', '') or '').strip())} --backend vllm --shutdown-backend"
            )
    if existing:
        base_url = normalize_openai_base_url(str(existing.get("base_url") or "").strip())
        if not base_url:
            raise RuntimeError(f"Detached vLLM control file is missing base_url: {control_file_path}")
        _wait_until_ready(process=None, base_url=base_url, timeout_s=float(args.vllm_timeout_s), api_key=api_key)
        resolved_model_id = _resolve_model_from_models_endpoint(
            base_url,
            timeout_s=float(args.vllm_timeout_s),
            api_key=api_key,
            served_model_name=(args.vllm_served_model_name or "").strip(),
        )
        logger = FileLogger.from_value(
            getattr(args, "vllm_log_file", ""),
            "backend",
            config_path=getattr(args, "_config_path", None),
        )
        tail_stop_event = threading.Event()
        tail_threads: list[threading.Thread] = []
        for source_key, source_name in (("engine_stdout_path", "engine_stdout"), ("engine_stderr_path", "engine_stderr")):
            path = str(existing.get(source_key) or "").strip()
            if not path or not os.path.isfile(path):
                continue
            start_offset = os.path.getsize(path)
            thread = threading.Thread(
                target=_FileTail(logger, source_name, path, start_offset, tail_stop_event).pump,
                daemon=True,
            )
            thread.start()
            tail_threads.append(thread)
        template_info = dict(existing.get("template_info") or {})
        tokenizer = None
        if AutoTokenizer is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
            except Exception:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
                except Exception:
                    tokenizer = None
        if tokenizer is not None and bool(template_info.get("chat_template_applied")):
            template_path = str(template_info.get("chat_template_requested") or "").strip()
            if template_path and os.path.isfile(template_path):
                try:
                    with open(template_path, "r", encoding="utf-8") as fh:
                        tokenizer.chat_template = fh.read()
                except Exception:
                    pass
        logger.log(f"attached_existing base_url={base_url} model={resolved_model_id}", source="backend")
        session = VLLMSession(
            args=args,
            process=None,
            launch_argv=list(existing.get("launch_argv") or []),
            base_url=base_url,
            resolved_model_id=resolved_model_id,
            tokenizer=tokenizer,
            api_key=api_key,
            template_info=template_info,
            logger=logger,
            control_file_path=control_file_path,
            managed_pid=existing_pid,
            managed_pgid=int(existing.get("pgid") or 0) or None,
            attached_existing=True,
            engine_stdout_start=os.path.getsize(str(existing.get("engine_stdout_path") or "")) if str(existing.get("engine_stdout_path") or "") and os.path.isfile(str(existing.get("engine_stdout_path") or "")) else 0,
            engine_stderr_start=os.path.getsize(str(existing.get("engine_stderr_path") or "")) if str(existing.get("engine_stderr_path") or "") and os.path.isfile(str(existing.get("engine_stderr_path") or "")) else 0,
            tail_stop_event=tail_stop_event,
            tail_threads=tail_threads,
        )
        session._engine_stdout_path = str(existing.get("engine_stdout_path") or "")
        session._engine_stderr_path = str(existing.get("engine_stderr_path") or "")
        return session

    port = requested_port
    if port == 0:
        port = _pick_free_port(host)
    base_url = normalize_openai_base_url((args.vllm_base_url or "").strip() or f"http://{host}:{port}")

    launch_argv = _build_launch_argv(args, host=host, port=port, model_id=model_id)
    requested_template = (args.chat_template or "").strip()
    template_requested_value = requested_template
    template_applied = False
    template_reason = "empty_default" if not requested_template else "unsupported_flag"
    if requested_template:
        cfg_path = (getattr(args, "_config_path", "") or "").strip()
        candidate = os.path.abspath(os.path.expanduser(requested_template))
        if cfg_path and not os.path.isabs(os.path.expanduser(requested_template)):
            candidate = os.path.abspath(os.path.join(os.path.dirname(cfg_path), requested_template))
        if os.path.isfile(candidate):
            template_requested_value = candidate
        if os.path.isfile(template_requested_value) and _supports_chat_template_flag(launch_argv):
            launch_argv.extend(["--chat-template", template_requested_value])
            template_applied = True
            template_reason = "applied"
    logger = FileLogger.from_value(
        getattr(args, "vllm_log_file", ""),
        "backend",
        config_path=getattr(args, "_config_path", None),
    )
    logger.log(f"launch_argv: {' '.join(shlex.quote(p) for p in launch_argv)}", source="app")

    stdout_path, stderr_path = _resolve_engine_log_paths(args)
    _ensure_parent_dir(stdout_path)
    _ensure_parent_dir(stderr_path)
    stdout_start = os.path.getsize(stdout_path) if os.path.isfile(stdout_path) else 0
    stderr_start = os.path.getsize(stderr_path) if os.path.isfile(stderr_path) else 0
    stdout_fh = open(stdout_path, "a", encoding="utf-8")
    stderr_fh = open(stderr_path, "a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            launch_argv,
            shell=False,
            stdout=stdout_fh,
            stderr=stderr_fh,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
    except OSError as exc:
        stdout_fh.close()
        stderr_fh.close()
        if port != 0 and ("address already in use" in str(exc).lower() or "98" in str(exc)):
            raise RuntimeError(
                f"Failed to launch vLLM on {host}:{port} (port busy). "
                f"Check: ss -ltnp | rg :{port}"
            ) from exc
        raise RuntimeError(f"Failed to launch vLLM process: {exc}") from exc
    finally:
        try:
            stdout_fh.close()
        except Exception:
            pass
        try:
            stderr_fh.close()
        except Exception:
            pass

    tail_stop_event = threading.Event()
    tail_threads: list[threading.Thread] = []
    for path, start_offset, source in (
        (stdout_path, stdout_start, "engine_stdout"),
        (stderr_path, stderr_start, "engine_stderr"),
    ):
        thread = threading.Thread(
            target=_FileTail(logger, source, path, start_offset, tail_stop_event).pump,
            daemon=True,
        )
        thread.start()
        tail_threads.append(thread)

    try:
        _wait_until_ready(process, base_url=base_url, timeout_s=float(args.vllm_timeout_s), api_key=api_key)
        resolved_model_id = _resolve_model_from_models_endpoint(
            base_url,
            timeout_s=float(args.vllm_timeout_s),
            api_key=api_key,
            served_model_name=(args.vllm_served_model_name or "").strip(),
        )
    except Exception as exc:
        tail_rows = logger.get_recent_logs(40)
        tail = "\n".join(tail_rows) if tail_rows else "(no captured logs)"
        lowered_tail = tail.lower()
        if process.poll() is not None and port != 0:
            if "address already in use" in lowered_tail or "already in use" in lowered_tail:
                raise RuntimeError(
                    f"Failed to launch vLLM on {host}:{port} (port busy). Check: ss -ltnp | rg :{port}"
                ) from exc
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGTERM)
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass
        raise RuntimeError(
            f"Failed to initialize vLLM managed server: {exc}\n"
            f"launch_argv: {' '.join(shlex.quote(p) for p in launch_argv)}\n"
            f"log_tail:\n{tail}"
        ) from exc

    logger.log(f"backend_ready base_url={base_url} model={resolved_model_id}", source="backend")
    tokenizer = None
    if AutoTokenizer is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
            except Exception:
                tokenizer = None
    if tokenizer is not None and template_applied and os.path.isfile(str(template_requested_value)):
        try:
            with open(str(template_requested_value), "r", encoding="utf-8") as fh:
                tokenizer.chat_template = fh.read()
        except Exception:
            pass

    try:
        pgid = os.getpgid(process.pid)
    except Exception:
        pgid = None
    template_info = {
        "template_control_level": "managed_server_template",
        "chat_template_requested": template_requested_value,
        "chat_template_applied": template_applied,
        "chat_template_reason": template_reason,
    }
    _write_control_file(
        control_file_path,
        {
            "backend": "vllm",
            "pid": process.pid,
            "pgid": pgid,
            "base_url": base_url,
            "resolved_model_id": resolved_model_id,
            "launch_argv": launch_argv,
            "signature": signature,
            "engine_stdout_path": stdout_path,
            "engine_stderr_path": stderr_path,
            "template_info": template_info,
            "created_at": time.time(),
        },
    )

    session = VLLMSession(
        args=args,
        process=process,
        launch_argv=launch_argv,
        base_url=base_url,
        resolved_model_id=resolved_model_id,
        tokenizer=tokenizer,
        api_key=api_key,
        template_info=template_info,
        logger=logger,
        control_file_path=control_file_path,
        managed_pid=process.pid,
        managed_pgid=pgid,
        attached_existing=False,
        engine_stdout_start=stdout_start,
        engine_stderr_start=stderr_start,
        tail_stop_event=tail_stop_event,
        tail_threads=tail_threads,
    )
    session._engine_stdout_path = stdout_path
    session._engine_stderr_path = stderr_path
    return session


def shutdown_managed_server(args: argparse.Namespace) -> dict[str, object]:
    control_file_path = _resolve_sidecar_path(args, "vllm-managed.json")
    existing = _load_control_file(control_file_path)
    if not existing:
        return {"stopped": False, "reason": "not_found", "control_file": control_file_path}
    pid = int(existing.get("pid") or 0)
    pgid = int(existing.get("pgid") or 0) or None
    if not _pid_alive(pid):
        _remove_control_file(control_file_path)
        return {"stopped": False, "reason": "stale", "control_file": control_file_path, "pid": pid}
    _stop_pid_group(pid, pgid)
    _remove_control_file(control_file_path)
    return {
        "stopped": True,
        "reason": "terminated",
        "control_file": control_file_path,
        "pid": pid,
        "base_url": existing.get("base_url"),
    }
