from __future__ import annotations

import argparse
import collections
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
from tui_app.events import Error
from tui_app.transports.openai_http import OpenAIHTTPSession, normalize_openai_base_url


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


class _StreamTail:
    def __init__(self, prefix: str, lines: collections.deque[str], tee_fh=None):
        self.prefix = prefix
        self.lines = lines
        self.tee_fh = tee_fh

    def pump(self, stream) -> None:
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                entry = f"[{self.prefix}] {line.rstrip()}"
                self.lines.append(entry)
                if self.tee_fh is not None:
                    try:
                        self.tee_fh.write(entry + "\n")
                        self.tee_fh.flush()
                    except Exception:
                        pass
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
        log_lines: collections.deque[str],
        log_file_handle=None,
    ):
        self.args = args
        self._process = process
        self._launch_argv = launch_argv
        self._base_url = base_url
        self._log_lines = log_lines
        self._closed = False
        self._log_file_handle = log_file_handle
        self._transport = OpenAIHTTPSession(
            args=args,
            resolved_model_id=resolved_model_id,
            base_url=base_url,
            api_key=api_key,
            timeout_s=float(args.vllm_timeout_s),
            backend_name="vllm",
        )
        self.resolved_model_id = resolved_model_id

    def describe(self) -> dict[str, object]:
        info = dict(self._transport.describe())
        info["managed_mode"] = True
        info["pid"] = self._process.pid
        info["launch_argv"] = " ".join(shlex.quote(part) for part in self._launch_argv)
        return info

    def get_recent_logs(self, n: int = 80) -> list[str]:
        size = max(1, int(n))
        data = list(self._log_lines)
        if size >= len(data):
            return data
        return data[-size:]

    def get_last_request(self) -> dict | None:
        getter = getattr(self._transport, "get_last_request", None)
        if callable(getter):
            return getter()
        return None

    def generate_turn(self, turn_id: int, messages: list[dict[str, object]], emit: EventEmitter) -> None:
        if self._process.poll() is not None:
            tail = "\n".join(self._log_lines) if self._log_lines else "(no captured logs)"
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
        self._transport.generate_turn(turn_id=turn_id, messages=messages, emit=emit)

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

        try:
            if self._log_file_handle is not None:
                self._log_file_handle.close()
        except Exception:
            pass


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
    logs: collections.deque[str] = collections.deque(maxlen=80)
    tee_handle = None
    log_file = (getattr(args, "vllm_log_file", "") or "").strip()
    if log_file:
        log_path = os.path.abspath(os.path.expanduser(log_file))
        parent = os.path.dirname(log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tee_handle = open(log_path, "a", encoding="utf-8")
        tee_handle.write(
            f"[managed_vllm] launch_argv: {' '.join(shlex.quote(p) for p in launch_argv)}\n"
        )
        tee_handle.flush()

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
        threading.Thread(target=_StreamTail("stdout", logs, tee_fh=tee_handle).pump, args=(process.stdout,), daemon=True).start()
    if process.stderr is not None:
        threading.Thread(target=_StreamTail("stderr", logs, tee_fh=tee_handle).pump, args=(process.stderr,), daemon=True).start()

    try:
        _wait_until_ready(process, base_url=base_url, timeout_s=float(args.vllm_timeout_s), api_key=api_key)
        resolved_model_id = _resolve_model_from_models_endpoint(
            base_url,
            timeout_s=float(args.vllm_timeout_s),
            api_key=api_key,
            served_model_name=(args.vllm_served_model_name or "").strip(),
        )
    except Exception as exc:
        tail = "\n".join(logs) if logs else "(no captured logs)"
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

    return VLLMSession(
        args=args,
        process=process,
        launch_argv=launch_argv,
        base_url=base_url,
        resolved_model_id=resolved_model_id,
        api_key=api_key,
        log_lines=logs,
        log_file_handle=tee_handle,
    )
