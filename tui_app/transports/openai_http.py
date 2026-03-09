from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request

from tui_app.backends.base import EventEmitter
from tui_app.events import AnswerDelta, Error, Finish, Meta, ThinkDelta, TurnRecord, TurnStart
from tui_app.log_file import FileLogger
from tui_app.think_router import ThinkRouter

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 20 * 1024 * 1024
MAX_CAPTURE_BYTES = 256 * 1024


def normalize_openai_base_url(base_url: str) -> str:
    raw = (base_url or "").strip().rstrip("/")
    if not raw:
        return ""
    if raw.endswith("/v1"):
        return raw
    return raw + "/v1"


def _looks_like_url(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith("http://") or lowered.startswith("https://") or lowered.startswith("data:")


def _build_data_url(path: str) -> tuple[str, int]:
    with open(path, "rb") as fh:
        raw = fh.read()
    size = len(raw)
    if size > MAX_IMAGE_BYTES:
        raise RuntimeError(
            f"Image too large: {path} ({size} bytes). Limit is {MAX_IMAGE_BYTES} bytes per image. "
            "Resize the image or use a hosted URL."
        )
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "application/octet-stream"
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}", size


def _sanitize_messages(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    clean: list[dict[str, object]] = []
    total_image_bytes = 0
    for msg in messages:
        role_obj = msg.get("role", "")
        role = str(role_obj).strip()
        if role not in {"system", "user", "assistant", "tool"}:
            continue

        content_obj = msg.get("content", "")
        text = content_obj if isinstance(content_obj, str) else str(content_obj)
        images_obj = msg.get("images")
        if not isinstance(images_obj, list) or not images_obj:
            clean.append({"role": role, "content": text})
            continue

        parts: list[dict[str, object]] = []
        if text:
            parts.append({"type": "text", "text": text})
        for entry in images_obj:
            if not isinstance(entry, str) or not entry.strip():
                continue
            value = entry.strip()
            if _looks_like_url(value):
                url = value
                size = 0
            else:
                if not os.path.isfile(value):
                    raise RuntimeError(f"Image path not found: {value}")
                url, size = _build_data_url(value)
            total_image_bytes += size
            if total_image_bytes > MAX_TOTAL_IMAGE_BYTES:
                raise RuntimeError(
                    f"Image payload too large ({total_image_bytes} bytes). Limit is {MAX_TOTAL_IMAGE_BYTES} bytes. "
                    "Attach fewer images, resize them, or use hosted URLs."
                )
            parts.append({"type": "image_url", "image_url": {"url": url}})
        clean.append({"role": role, "content": parts if parts else text})
    return clean


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def _sanitize_for_capture(obj):
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            key_l = str(key).lower()
            if key_l in {"authorization", "api_key", "openai_api_key", "vllm_api_key"}:
                out[key] = "***"
                continue
            out[key] = _sanitize_for_capture(value)
        return out
    if isinstance(obj, list):
        return [_sanitize_for_capture(x) for x in obj]
    if isinstance(obj, str):
        if obj.startswith("data:"):
            mime = "unknown"
            if ";" in obj and obj.startswith("data:"):
                mime = obj[5 : obj.find(";")] or "unknown"
            approx_bytes = int((len(obj) - obj.find(",") - 1) * 0.75) if "," in obj else 0
            return {"url": "(data-url omitted)", "mime": mime, "bytes": approx_bytes}
        if "bearer " in obj.lower():
            return "***"
    return obj


def _truncate_capture(data: dict) -> dict:
    raw = json.dumps(data, ensure_ascii=False, default=str)
    if len(raw.encode("utf-8")) <= MAX_CAPTURE_BYTES:
        return data
    marker = "...(truncated)"
    keep_bytes = max(1024, MAX_CAPTURE_BYTES - len(marker.encode("utf-8")) - 64)
    trimmed = raw.encode("utf-8")[:keep_bytes].decode("utf-8", errors="ignore")
    return {"truncated": True, "max_bytes": MAX_CAPTURE_BYTES, "data": trimmed + marker}


def _json_request(
    method: str,
    url: str,
    *,
    timeout: float,
    api_key: str = "",
    payload: dict | None = None,
):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, method=method, data=data, headers=headers)
    return urllib.request.urlopen(req, timeout=timeout)


def resolve_model_once(base_url: str, *, timeout_s: float, api_key: str) -> str:
    resolved_base = normalize_openai_base_url(base_url)
    url = _join_url(resolved_base, "/models")
    with _json_request("GET", url, timeout=max(1.0, min(10.0, timeout_s)), api_key=api_key) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    data = json.loads(raw)
    models = data.get("data")
    if not isinstance(models, list):
        raise RuntimeError("Model resolution failed: /v1/models response missing data[]")
    ids: list[str] = []
    for item in models:
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip():
            ids.append(item["id"].strip())
    if len(ids) == 1:
        return ids[0]
    if not ids:
        raise RuntimeError("Model resolution failed: /v1/models returned no models")
    raise RuntimeError(
        "Model resolution failed: server returned multiple models. "
        f"Set model_id explicitly. Available: {', '.join(ids)}"
    )


class OpenAIHTTPSession:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        resolved_model_id: str,
        base_url: str,
        api_key: str,
        timeout_s: float,
        backend_name: str,
        logger: FileLogger | None = None,
    ):
        self.args = args
        self.resolved_model_id = resolved_model_id
        self.base_url = normalize_openai_base_url(base_url)
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.backend_name = backend_name
        self._ignored_knobs_once = False
        self.logger = logger
        self._last_request: dict | None = None

    def close(self) -> None:
        if self.logger is not None:
            self.logger.close()

    def get_recent_logs(self, n: int = 80) -> list[str]:
        if self.logger is None:
            return []
        return self.logger.get_recent_logs(n=n)

    def get_last_request(self) -> dict | None:
        return self._last_request

    def describe(self) -> dict[str, object]:
        return {
            "base_url": self.base_url,
            "model_id": self.resolved_model_id,
            "api_key": "(set)" if self.api_key else "(unset)",
            "timeout_s": self.timeout_s,
        }

    def _emit_ignored_knobs_once(self, emit: EventEmitter, turn_id: int) -> None:
        if self._ignored_knobs_once:
            return
        ignored: list[str] = []
        if self.backend_name != "vllm" and self.args.top_k is not None:
            ignored.append("top_k")
        if self.backend_name != "vllm" and self.args.min_p is not None:
            ignored.append("min_p")
        if self.args.typical_p is not None:
            ignored.append("typical_p")
        if self.backend_name != "vllm" and self.args.repetition_penalty not in (None, 1.0):
            ignored.append("repetition_penalty")
        if self.backend_name != "vllm" and self.args.presence_penalty is not None:
            ignored.append("presence_penalty")
        if self.backend_name != "vllm" and self.args.frequency_penalty is not None:
            ignored.append("frequency_penalty")
        if ignored:
            emit(Meta(turn_id=turn_id, key="ignored_knobs", value=ignored))
        self._ignored_knobs_once = True

    def generate_turn(self, turn_id: int, messages: list[dict[str, object]], emit: EventEmitter) -> None:
        emit(TurnStart(turn_id=turn_id))
        started = time.time()
        deadline = started + max(1.0, self.timeout_s)
        self._emit_ignored_knobs_once(emit, turn_id)
        if self.logger is not None:
            self.logger.log(
                f"turn_start id={turn_id} messages={len(messages)} model={self.resolved_model_id} "
                f"max_new_tokens={self.args.max_new_tokens}"
            )

        router = ThinkRouter(assume_think=self.args.assume_think)
        raw_parts: list[str] = []
        think_parts: list[str] = []
        answer_parts: list[str] = []
        tool_calls: dict[str, dict[str, str | int]] = {}
        stage = "build_payload"

        def _emit_think(text: str) -> None:
            if not text:
                return
            think_parts.append(text)
            emit(ThinkDelta(turn_id=turn_id, text=text))
            token_inc = len(re.findall(r"\S+", text))
            if token_inc > 0:
                emit(Meta(turn_id=turn_id, key="think_tokens_inc", value=token_inc))

        try:
            payload_messages = _sanitize_messages(messages)
            payload: dict[str, object] = {
                "messages": payload_messages,
                "stream": True,
                "max_tokens": self.args.max_new_tokens,
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
            }
            if self.resolved_model_id:
                payload["model"] = self.resolved_model_id
            if self.args.stop_strings:
                payload["stop"] = self.args.stop_strings
            if self.args.seed is not None:
                payload["seed"] = self.args.seed
            if self.backend_name == "vllm":
                if self.args.presence_penalty not in (None, 0.0):
                    payload["presence_penalty"] = self.args.presence_penalty
                if self.args.frequency_penalty not in (None, 0.0):
                    payload["frequency_penalty"] = self.args.frequency_penalty
                if self.args.top_k not in (None, -1):
                    payload["top_k"] = self.args.top_k
                if self.args.min_p not in (None, 0.0):
                    payload["min_p"] = self.args.min_p
                if self.args.repetition_penalty not in (None, 1.0):
                    payload["repetition_penalty"] = self.args.repetition_penalty
                if self.args.stop_token_ids:
                    payload["stop_token_ids"] = self.args.stop_token_ids
                if self.args.ignore_eos is True:
                    payload["ignore_eos"] = self.args.ignore_eos
                if self.args.min_tokens not in (None, 0):
                    payload["min_tokens"] = self.args.min_tokens
                if self.args.best_of not in (None, 1):
                    payload["best_of"] = self.args.best_of
                if self.args.use_beam_search is True:
                    payload["use_beam_search"] = self.args.use_beam_search
                if self.args.length_penalty not in (None, 1.0):
                    payload["length_penalty"] = self.args.length_penalty
                if self.args.include_stop_str_in_output is True:
                    payload["include_stop_str_in_output"] = self.args.include_stop_str_in_output
                if self.args.skip_special_tokens is False:
                    payload["skip_special_tokens"] = self.args.skip_special_tokens
                if self.args.spaces_between_special_tokens is False:
                    payload["spaces_between_special_tokens"] = self.args.spaces_between_special_tokens
                if self.args.truncate_prompt_tokens not in (None, 0):
                    payload["truncate_prompt_tokens"] = self.args.truncate_prompt_tokens
                if self.args.allowed_token_ids:
                    payload["allowed_token_ids"] = self.args.allowed_token_ids
                if self.args.prompt_logprobs not in (None, 0):
                    payload["prompt_logprobs"] = self.args.prompt_logprobs

            stage = "request_open"
            url = _join_url(self.base_url, "/chat/completions")
            if getattr(self.args, "capture_last_request", False):
                captured = {
                    "backend": self.backend_name,
                    "url": url,
                    "payload": _sanitize_for_capture(payload),
                    "api_key": "***" if self.api_key else "(unset)",
                }
                self._last_request = _truncate_capture(captured)
            req_timeout = max(1.0, self.timeout_s)
            with _json_request("POST", url, timeout=req_timeout, api_key=self.api_key, payload=payload) as resp:
                stage = "stream_parse"
                buffer = ""
                for raw_line in resp:
                    if time.time() > deadline:
                        raise RuntimeError(f"Request timed out after {self.timeout_s}s (total timeout).")
                    line = raw_line.decode("utf-8", errors="replace")
                    buffer += line
                    while "\n" in buffer:
                        current, buffer = buffer.split("\n", 1)
                        current = current.strip()
                        if not current or not current.startswith("data:"):
                            continue
                        data_part = current[5:].strip()
                        if data_part == "[DONE]":
                            buffer = ""
                            raise StopIteration
                        frame = json.loads(data_part)
                        if isinstance(frame, dict) and isinstance(frame.get("error"), dict):
                            err = frame["error"].get("message") or str(frame["error"])
                            raise RuntimeError(f"Server error frame: {err}")

                        choices = frame.get("choices")
                        if not isinstance(choices, list) or not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        if not isinstance(delta, dict):
                            continue

                        reasoning = delta.get("reasoning")
                        if not isinstance(reasoning, str):
                            reasoning = delta.get("reasoning_content")
                        if isinstance(reasoning, str) and reasoning:
                            raw_parts.append(reasoning)
                            _emit_think(reasoning)

                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            raw_parts.append(content)
                            if isinstance(reasoning, str) and reasoning:
                                answer_parts.append(content)
                                emit(AnswerDelta(turn_id=turn_id, text=content))
                            else:
                                for channel, text in router.feed(content):
                                    if channel == "think":
                                        _emit_think(text)
                                    else:
                                        answer_parts.append(text)
                                        emit(AnswerDelta(turn_id=turn_id, text=text))

                        tcs = delta.get("tool_calls")
                        if isinstance(tcs, list):
                            for tc in tcs:
                                if not isinstance(tc, dict):
                                    continue
                                idx = tc.get("index", 0)
                                idx_key = str(idx)
                                rec = tool_calls.setdefault(
                                    idx_key,
                                    {"index": idx, "id": "", "name": "", "arguments": ""},
                                )
                                tc_id = tc.get("id")
                                if isinstance(tc_id, str) and tc_id:
                                    rec["id"] = tc_id
                                fn = tc.get("function")
                                if isinstance(fn, dict):
                                    name = fn.get("name")
                                    if isinstance(name, str) and name:
                                        rec["name"] = name
                                    args_part = fn.get("arguments")
                                    if isinstance(args_part, str) and args_part:
                                        rec["arguments"] = str(rec.get("arguments", "")) + args_part

        except StopIteration:
            pass
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore").strip()
            detail = f"OpenAI-compatible API request failed: HTTP {exc.code} {exc.reason}"
            if body:
                detail += f" | body: {body}"
            if self.logger is not None:
                self.logger.log(f"turn_error id={turn_id} stage={stage} error={detail}")
            emit(Error(turn_id=turn_id, message=detail))
            return
        except urllib.error.URLError as exc:
            if self.logger is not None:
                self.logger.log(f"turn_error id={turn_id} stage={stage} error={exc}")
            emit(Error(turn_id=turn_id, message=f"OpenAI-compatible API request failed: {exc}"))
            return
        except Exception as exc:
            if self.logger is not None:
                self.logger.log(f"turn_error id={turn_id} stage={stage} error={exc}")
            emit(Error(turn_id=turn_id, message=f"{stage}: {exc}"))
            return

        for channel, text in router.flush():
            if channel == "think":
                _emit_think(text)
            else:
                answer_parts.append(text)
                emit(AnswerDelta(turn_id=turn_id, text=text))

        tool_calls_out: list[dict[str, object]] = []
        if tool_calls:
            for key in sorted(tool_calls.keys(), key=lambda x: int(x)):
                item = tool_calls[key]
                tool_calls_out.append(
                    {
                        "index": item.get("index"),
                        "id": item.get("id") or "",
                        "name": item.get("name") or "",
                        "arguments": item.get("arguments") or "",
                    }
                )
            raw_parts.append("\n\n[tool_calls]\n")
            raw_parts.append(json.dumps(tool_calls_out, ensure_ascii=False))

        ended = time.time()
        gen = {
            "max_tokens": self.args.max_new_tokens,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "stop": self.args.stop_strings,
            "seed": self.args.seed,
        }
        if self.backend_name == "vllm":
            gen["top_k"] = self.args.top_k
            gen["min_p"] = self.args.min_p
            gen["repetition_penalty"] = self.args.repetition_penalty
            gen["presence_penalty"] = self.args.presence_penalty
            gen["frequency_penalty"] = self.args.frequency_penalty
            gen["stop_token_ids"] = self.args.stop_token_ids
            gen["ignore_eos"] = self.args.ignore_eos
            gen["min_tokens"] = self.args.min_tokens
            gen["best_of"] = self.args.best_of
            gen["use_beam_search"] = self.args.use_beam_search
            gen["length_penalty"] = self.args.length_penalty
            gen["include_stop_str_in_output"] = self.args.include_stop_str_in_output
            gen["skip_special_tokens"] = self.args.skip_special_tokens
            gen["spaces_between_special_tokens"] = self.args.spaces_between_special_tokens
            gen["truncate_prompt_tokens"] = self.args.truncate_prompt_tokens
            gen["allowed_token_ids"] = self.args.allowed_token_ids
            gen["prompt_logprobs"] = self.args.prompt_logprobs
        if tool_calls_out:
            gen["tool_calls"] = tool_calls_out
        record = TurnRecord(
            raw="".join(raw_parts),
            think="".join(think_parts),
            answer="".join(answer_parts).strip(),
            ended_in_think=(router.mode == "think"),
            backend=self.backend_name,
            model_id=self.resolved_model_id,
            gen=gen,
            timing={"start": started, "end": ended, "elapsed": max(0.0, ended - started)},
            trimmed_messages=messages,
        )
        emit(Finish(turn_id=turn_id, record=record))
        if self.logger is not None:
            self.logger.log(
                f"turn_finish id={turn_id} elapsed_s={record.timing.get('elapsed', 0):.3f} "
                f"think_chars={len(record.think)} answer_chars={len(record.answer)}"
            )
