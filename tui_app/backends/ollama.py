from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

from tui_app.backends.base import EventEmitter
from tui_app.events import AnswerDelta, Error, Finish, ThinkDelta, TurnRecord, TurnStart
from tui_app.think_router import ThinkRouter


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

    def __init__(self, args: argparse.Namespace, resolved_model_id: str, host: str):
        self.args = args
        self.resolved_model_id = resolved_model_id
        self.host = host

    def generate_turn(self, turn_id: int, messages: list[dict[str, str]], emit: EventEmitter) -> None:
        emit(TurnStart(turn_id=turn_id))
        started = time.time()

        payload = {
            "model": self.resolved_model_id,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
                "top_k": self.args.top_k,
                "num_predict": self.args.max_new_tokens,
                "stop": self.args.stop_strings,
            },
        }
        payload["options"] = {k: v for k, v in payload["options"].items() if v is not None}
        if self.args.ollama_think in {"true", "false"}:
            payload["think"] = self.args.ollama_think == "true"

        req = urllib.request.Request(
            self.host.rstrip("/") + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        router = ThinkRouter()
        raw_parts: list[str] = []
        think_parts: list[str] = []
        answer_parts: list[str] = []

        try:
            with urllib.request.urlopen(req, timeout=self.args.ollama_timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    msg = obj.get("message", {}) or {}

                    thinking = msg.get("thinking", "")
                    content = msg.get("content", "")
                    if not isinstance(thinking, str):
                        thinking = str(thinking)
                    if not isinstance(content, str):
                        content = str(content)

                    if thinking:
                        think_parts.append(thinking)
                        raw_parts.append(thinking)
                        emit(ThinkDelta(turn_id=turn_id, text=thinking))

                    if content:
                        raw_parts.append(content)
                        if thinking:
                            answer_parts.append(content)
                            emit(AnswerDelta(turn_id=turn_id, text=content))
                        else:
                            for channel, text in router.feed(content):
                                if channel == "think":
                                    think_parts.append(text)
                                    emit(ThinkDelta(turn_id=turn_id, text=text))
                                else:
                                    answer_parts.append(text)
                                    emit(AnswerDelta(turn_id=turn_id, text=text))

                    if obj.get("done"):
                        break
        except urllib.error.URLError as exc:
            emit(Error(turn_id=turn_id, message=f"Ollama API request failed: {exc}"))
            return
        except Exception as exc:
            emit(Error(turn_id=turn_id, message=str(exc)))
            return

        for channel, text in router.flush():
            if channel == "think":
                think_parts.append(text)
                emit(ThinkDelta(turn_id=turn_id, text=text))
            else:
                answer_parts.append(text)
                emit(AnswerDelta(turn_id=turn_id, text=text))

        ended = time.time()
        record = TurnRecord(
            raw="".join(raw_parts),
            think="".join(think_parts),
            answer="".join(answer_parts).strip(),
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
            trimmed_messages=messages,
        )
        emit(Finish(turn_id=turn_id, record=record))


def create_session(args: argparse.Namespace) -> OllamaSession:
    model_name = args.model_id
    if model_name.startswith("ollama:"):
        model_name = model_name.split(":", 1)[1]
    host = resolve_host(args.ollama_host, args.ollama_timeout)
    print(f"Using Ollama host: {host}")
    return OllamaSession(args=args, resolved_model_id=model_name, host=host)
