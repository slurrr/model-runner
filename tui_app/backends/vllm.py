from __future__ import annotations

import argparse
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
    probe.append("--help")
    try:
        out = subprocess.check_output(probe, text=True, stderr=subprocess.STDOUT, timeout=8)
    except Exception:
        return False
    return "--chat-template" in out


class _StreamTail:
    def __init__(self, logger: FileLogger, source: str):
        self.logger = logger
        self.source = source

    def pump(self, stream) -> None:
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                self.logger.log(line.rstrip(), source=self.source)
        except Exception:
            return
        finally:
            try:
                stream.close()
            except Exception:
                pass


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
        tokenizer=None,
    ):
        self.args = args
        self._process = process
        self._launch_argv = launch_argv
        self._base_url = base_url
        self.logger = logger
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
        info["pid"] = self._process.pid
        info["launch_argv"] = " ".join(shlex.quote(part) for part in self._launch_argv)
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
        if self._process.poll() is not None:
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
        if self._process.poll() is None:
            try:
                pgid = os.getpgid(self._process.pid)
                os.killpg(pgid, signal.SIGTERM)
            except Exception:
                try:
                    self._process.terminate()
                except Exception:
                    pass
            try:
                self._process.wait(timeout=5.0)
            except Exception:
                try:
                    pgid = os.getpgid(self._process.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except Exception:
                    try:
                        self._process.kill()
                    except Exception:
                        pass
                try:
                    self._process.wait(timeout=2.0)
                except Exception:
                    pass

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

    extra = args.vllm_extra_args or []
    if isinstance(extra, str):
        extra = shlex.split(extra)
    if isinstance(extra, list):
        launch.extend([str(x) for x in extra if str(x).strip()])
    if "--enable-force-include-usage" not in launch:
        launch.append("--enable-force-include-usage")
    return launch


def _wait_until_ready(
    process: subprocess.Popen,
    *,
    base_url: str,
    timeout_s: float,
    api_key: str,
) -> None:
    deadline = time.time() + max(5.0, min(120.0, timeout_s))
    err_last = ""
    while time.time() < deadline:
        if process.poll() is not None:
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
            if cfg_model_id:
                model_id = apply_machine_model_root(cfg_model_id)
        except Exception:
            pass
    if not model_id:
        raise RuntimeError("vLLM managed mode requires model_id (path or model name).")

    host = (args.vllm_host or "127.0.0.1").strip()
    port = int(args.vllm_port)
    if port < 0:
        raise RuntimeError("vLLM port must be >= 0.")
    if port == 0:
        port = _pick_free_port(host)
    base_url = normalize_openai_base_url((args.vllm_base_url or "").strip() or f"http://{host}:{port}")
    api_key = _resolve_api_key(args)

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

    try:
        process = subprocess.Popen(
            launch_argv,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
    except OSError as exc:
        if port != 0 and ("address already in use" in str(exc).lower() or "98" in str(exc)):
            raise RuntimeError(
                f"Failed to launch vLLM on {host}:{port} (port busy). "
                f"Check: ss -ltnp | rg :{port}"
            ) from exc
        raise RuntimeError(f"Failed to launch vLLM process: {exc}") from exc

    if process.stdout is not None:
        threading.Thread(target=_StreamTail(logger, "engine_stdout").pump, args=(process.stdout,), daemon=True).start()
    if process.stderr is not None:
        threading.Thread(target=_StreamTail(logger, "engine_stderr").pump, args=(process.stderr,), daemon=True).start()

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

    return VLLMSession(
        args=args,
        process=process,
        launch_argv=launch_argv,
        base_url=base_url,
        resolved_model_id=resolved_model_id,
        tokenizer=tokenizer,
        api_key=api_key,
        template_info={
            "template_control_level": "managed_server_template",
            "chat_template_requested": template_requested_value,
            "chat_template_applied": template_applied,
            "chat_template_reason": template_reason,
        },
        logger=logger,
    )
