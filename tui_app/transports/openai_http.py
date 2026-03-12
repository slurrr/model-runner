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
from copy import deepcopy

from tui_app.backends.base import EventEmitter
from tui_app.context_policy import (
    build_context_limit_error,
    build_retry_report,
    drop_oldest_history_message,
    is_context_overflow_text,
    reserve_generation_tokens,
)
from tui_app.events import AnswerDelta, Error, Finish, Meta, ThinkDelta, TurnRecord, TurnStart
from tui_app.history_control import append_assistant_history
from tui_app.knobs import SUPPORTED_KNOBS, finalize_knob_report, unsupported_user_set
from tui_app.log_file import FileLogger
from tui_app.think_router import ThinkRouter
from tui_app.tools import build_tool_runtime

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

        content_obj = msg.get("content")
        text: str | None
        if content_obj is None:
            text = None
        else:
            text = content_obj if isinstance(content_obj, str) else str(content_obj)
        images_obj = msg.get("images")
        if not isinstance(images_obj, list) or not images_obj:
            out: dict[str, object] = {"role": role, "content": text}
            tool_calls = msg.get("tool_calls")
            if role == "assistant" and isinstance(tool_calls, list):
                out["tool_calls"] = deepcopy(tool_calls)
            tool_call_id = msg.get("tool_call_id")
            if role == "tool" and isinstance(tool_call_id, str) and tool_call_id.strip():
                out["tool_call_id"] = tool_call_id.strip()
            clean.append(out)
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
        out = {"role": role, "content": parts if parts else text}
        clean.append(out)
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
        template_info: dict[str, object] | None = None,
        logger: FileLogger | None = None,
    ):
        self.args = args
        self.resolved_model_id = resolved_model_id
        self.base_url = normalize_openai_base_url(base_url)
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.backend_name = backend_name
        self._ignored_knobs_once = False
        self.template_info = dict(template_info or {})
        self.logger = logger
        self._last_request: dict | None = None
        self._tool_runtime = build_tool_runtime(args, backend_name)

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
            "base_url": self.base_url,
            "model_id": self.resolved_model_id,
            "api_key": "(set)" if self.api_key else "(unset)",
            "timeout_s": self.timeout_s,
            **self.template_info,
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

    def generate_turn(
        self,
        turn_id: int,
        messages: list[dict[str, object]],
        emit: EventEmitter,
        *,
        emit_turn_start: bool = True,
    ) -> None:
        class _ContextOverflow(RuntimeError):
            pass

        if emit_turn_start:
            emit(TurnStart(turn_id=turn_id))
        started = time.time()
        deadline = started + max(1.0, self.timeout_s)
        self._emit_ignored_knobs_once(emit, turn_id)
        if self.logger is not None:
            self.logger.log(
                f"turn_start id={turn_id} messages={len(messages)} model={self.resolved_model_id} "
                f"max_new_tokens={self.args.max_new_tokens}",
                source="transport",
            )

        context_window = None
        if self.backend_name == "vllm" and int(self.args.vllm_max_model_len or 0) > 0:
            context_window = int(self.args.vllm_max_model_len)
        reserved_generation = reserve_generation_tokens(context_window, self.args.max_new_tokens)
        working_messages = list(messages)
        overflow_retries = 0
        dropped_roles: list[str] = []
        tool_runtime = self._tool_runtime
        tools_active = (
            tool_runtime.enabled
            and tool_runtime.supported_backend
            and bool(tool_runtime.tool_names())
        )
        tool_activity: list[dict[str, object]] = []
        all_tool_calls_for_gen: list[dict[str, object]] = []
        aggregate_prompt_tokens = 0
        aggregate_completion_tokens = 0
        aggregate_total_tokens = 0
        final_finish_reason: str | None = None
        final_knob_report: dict[str, object] | None = None
        final_context_prompt_tokens: int | None = None
        final_answer_text = ""
        final_think_text = ""
        ended_in_think = False
        aggregate_raw_parts: list[str] = []
        aggregate_think_parts: list[str] = []
        aggregate_answer_parts: list[str] = []

        def _drop_oldest_turn() -> bool:
            dropped_chunk = drop_oldest_history_message(working_messages)
            if dropped_chunk is None:
                return False
            dropped_roles.extend(dropped_chunk)
            return True

        def _run_attempt(*, include_tools: bool):
            router = ThinkRouter(assume_think=self.args.assume_think)
            raw_parts: list[str] = []
            think_parts: list[str] = []
            answer_parts: list[str] = []
            tool_calls: dict[str, dict[str, str | int]] = {}
            usage_totals: dict[str, int] | None = None
            finish_reason: str | None = None
            stage = "build_payload"

            def _emit_generated_text(text: str) -> None:
                if not text:
                    return
                token_inc = len(re.findall(r"\S+", text))
                if token_inc > 0:
                    emit(Meta(turn_id=turn_id, key="generated_tokens_inc", value=token_inc))

            def _emit_think(text: str) -> None:
                if not text:
                    return
                think_parts.append(text)
                emit(ThinkDelta(turn_id=turn_id, text=text))

            payload_messages = _sanitize_messages(working_messages)
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
            if include_tools:
                payload["tools"] = tool_runtime.exposed_schema()
            if self.backend_name == "vllm":
                payload["stream_options"] = {"include_usage": True}
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

            knob_sent = {
                "max_new_tokens": payload["max_tokens"],
                "temperature": payload["temperature"],
                "top_p": payload["top_p"],
            }
            if "stop" in payload:
                knob_sent["stop_strings"] = payload["stop"]
            if "seed" in payload:
                knob_sent["seed"] = payload["seed"]
            if self.backend_name == "vllm":
                payload_to_knob = {
                    "top_k": "top_k",
                    "min_p": "min_p",
                    "repetition_penalty": "repetition_penalty",
                    "presence_penalty": "presence_penalty",
                    "frequency_penalty": "frequency_penalty",
                    "stop_token_ids": "stop_token_ids",
                    "ignore_eos": "ignore_eos",
                    "min_tokens": "min_tokens",
                    "best_of": "best_of",
                    "use_beam_search": "use_beam_search",
                    "length_penalty": "length_penalty",
                    "include_stop_str_in_output": "include_stop_str_in_output",
                    "skip_special_tokens": "skip_special_tokens",
                    "spaces_between_special_tokens": "spaces_between_special_tokens",
                    "truncate_prompt_tokens": "truncate_prompt_tokens",
                    "allowed_token_ids": "allowed_token_ids",
                    "prompt_logprobs": "prompt_logprobs",
                }
                for payload_key, knob_key in payload_to_knob.items():
                    if payload_key in payload:
                        knob_sent[knob_key] = payload[payload_key]
            knob_report = finalize_knob_report(
                sent=knob_sent,
                supported=SUPPORTED_KNOBS[self.backend_name],
                ignored=unsupported_user_set(self.args, self.backend_name),
            )

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
            try:
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
                                if not raw_parts and not think_parts and not answer_parts and is_context_overflow_text(err):
                                    raise _ContextOverflow(str(err))
                                raise RuntimeError(f"Server error frame: {err}")

                            usage = frame.get("usage")
                            if isinstance(usage, dict):
                                prompt_tokens = usage.get("prompt_tokens")
                                completion_tokens = usage.get("completion_tokens")
                                total_tokens = usage.get("total_tokens")
                                counts: dict[str, int] = {}
                                if isinstance(prompt_tokens, int):
                                    counts["prompt_tokens"] = int(prompt_tokens)
                                if isinstance(completion_tokens, int):
                                    counts["completion_tokens"] = int(completion_tokens)
                                if isinstance(total_tokens, int):
                                    counts["total_tokens"] = int(total_tokens)
                                if "total_tokens" not in counts and "prompt_tokens" in counts and "completion_tokens" in counts:
                                    counts["total_tokens"] = counts["prompt_tokens"] + counts["completion_tokens"]
                                if counts:
                                    usage_totals = counts
                                    for key, value in counts.items():
                                        emit(Meta(turn_id=turn_id, key=key, value=value))

                            choices = frame.get("choices")
                            if not isinstance(choices, list) or not choices:
                                continue
                            choice0 = choices[0]
                            choice_finish_reason = choice0.get("finish_reason")
                            if isinstance(choice_finish_reason, str) and choice_finish_reason:
                                finish_reason = choice_finish_reason
                            delta = choice0.get("delta") or {}
                            if not isinstance(delta, dict):
                                continue

                            reasoning = delta.get("reasoning")
                            if not isinstance(reasoning, str):
                                reasoning = delta.get("reasoning_content")
                            if isinstance(reasoning, str) and reasoning:
                                _emit_generated_text(reasoning)
                                raw_parts.append(reasoning)
                                _emit_think(reasoning)

                            content = delta.get("content")
                            if isinstance(content, str) and content:
                                _emit_generated_text(content)
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
                                        {"index": idx, "id": "", "type": "function", "name": "", "arguments": ""},
                                    )
                                    tc_id = tc.get("id")
                                    if isinstance(tc_id, str) and tc_id:
                                        rec["id"] = tc_id
                                    tc_type = tc.get("type")
                                    if isinstance(tc_type, str) and tc_type:
                                        rec["type"] = tc_type
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
                if not raw_parts and not think_parts and not answer_parts and is_context_overflow_text(detail):
                    raise _ContextOverflow(detail)
                raise RuntimeError(detail) from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"OpenAI-compatible API request failed: {exc}") from exc
            except _ContextOverflow:
                raise
            except Exception as exc:
                if not raw_parts and not think_parts and not answer_parts and is_context_overflow_text(str(exc)):
                    raise _ContextOverflow(str(exc)) from exc
                raise RuntimeError(f"{stage}: {exc}") from exc

            for channel, text in router.flush():
                if channel == "think":
                    _emit_think(text)
                else:
                    answer_parts.append(text)
                    emit(AnswerDelta(turn_id=turn_id, text=text))

            tool_calls_out: list[dict[str, object]] = []
            assistant_tool_calls: list[dict[str, object]] = []
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
                    assistant_tool_calls.append(
                        {
                            "id": item.get("id") or "",
                            "type": item.get("type") or "function",
                            "function": {
                                "name": item.get("name") or "",
                                "arguments": item.get("arguments") or "",
                            },
                        }
                    )
            answer_text = "".join(answer_parts)
            assistant_message: dict[str, object] | None = None
            if assistant_tool_calls:
                assistant_message = {
                    "role": "assistant",
                    "content": answer_text if answer_text else None,
                    "tool_calls": assistant_tool_calls,
                }
            return (
                payload,
                knob_report,
                raw_parts,
                think_parts,
                answer_parts,
                usage_totals,
                finish_reason,
                tool_calls_out,
                assistant_message,
                router.mode == "think",
            )

        while True:
            try:
                (
                    payload,
                    knob_report,
                    raw_parts,
                    think_parts,
                    answer_parts,
                    usage_totals,
                    finish_reason,
                    tool_calls_out,
                    assistant_message,
                    ended_in_think,
                ) = _run_attempt(include_tools=tools_active and tool_runtime.max_calls_per_turn > len(all_tool_calls_for_gen))
            except _ContextOverflow:
                if _drop_oldest_turn():
                    overflow_retries += 1
                    continue
                detail = build_context_limit_error(
                    build_retry_report(
                        messages,
                        working_messages,
                        strategy="overflow_retry",
                        context_window=context_window,
                        reserved_generation_tokens=reserved_generation,
                        overflow_retries=overflow_retries,
                        fit=False,
                        prompt_tokens=None,
                        dropped_roles=dropped_roles,
                    )
                )
                if self.logger is not None:
                    self.logger.log(f"turn_error id={turn_id} error={detail}", source="transport")
                emit(Error(turn_id=turn_id, message=detail))
                return
            except Exception as exc:
                if self.logger is not None:
                    self.logger.log(f"turn_error id={turn_id} error={exc}", source="transport")
                emit(Error(turn_id=turn_id, message=str(exc)))
                return

            final_knob_report = knob_report
            final_finish_reason = finish_reason
            final_answer_text = "".join(answer_parts)
            final_think_text = "".join(think_parts)
            aggregate_raw_parts.extend(raw_parts)
            aggregate_think_parts.extend(think_parts)
            aggregate_answer_parts.extend(answer_parts)
            if isinstance(usage_totals, dict):
                prompt_tokens = usage_totals.get("prompt_tokens")
                completion_tokens = usage_totals.get("completion_tokens")
                total_tokens = usage_totals.get("total_tokens")
                if isinstance(prompt_tokens, int):
                    aggregate_prompt_tokens += prompt_tokens
                    final_context_prompt_tokens = prompt_tokens
                if isinstance(completion_tokens, int):
                    aggregate_completion_tokens += completion_tokens
                if isinstance(total_tokens, int):
                    aggregate_total_tokens += total_tokens
                elif isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                    aggregate_total_tokens += prompt_tokens + completion_tokens

            if tool_calls_out:
                all_tool_calls_for_gen.extend(tool_calls_out)

            if not (tools_active and tool_calls_out):
                break

            if assistant_message is not None:
                working_messages.append(assistant_message)

            remaining_calls = max(0, tool_runtime.max_calls_per_turn - len(tool_activity))
            for index, call in enumerate(tool_calls_out):
                tool_call_id = str(call.get("id", "") or "").strip() or f"tool_call_{len(tool_activity) + 1}"
                name = str(call.get("name", "") or "").strip()
                arguments_raw = str(call.get("arguments", "") or "")
                denied_by_policy = False
                if name in tool_runtime.available_tools and name not in tool_runtime.exposed_tools:
                    denied_by_policy = True

                activity: dict[str, object]
                if tool_runtime.mode == "dry_run":
                    activity = tool_runtime.execute(
                        name=name,
                        arguments_raw=arguments_raw,
                        denied_by_policy=denied_by_policy,
                    )
                    activity["tool_call_id"] = tool_call_id
                elif index >= remaining_calls:
                    limit_msg = "Tool execution limit reached for this turn."
                    activity = {
                        "tool_call_id": tool_call_id,
                        "name": name,
                        "arguments_raw": arguments_raw,
                        "arguments_json": None,
                        "status": "max_calls_exceeded",
                        "result": limit_msg,
                        "error": limit_msg,
                    }
                    activity["result"] = str(activity["result"])[: tool_runtime.max_result_chars]
                else:
                    if self.logger is not None:
                        self.logger.log(
                            f"tool_call_execute_start id={turn_id} tool_call_id={tool_call_id} name={name}",
                            source="app",
                        )
                    activity = tool_runtime.execute(
                        name=name,
                        arguments_raw=arguments_raw,
                        denied_by_policy=denied_by_policy,
                    )
                    activity["tool_call_id"] = tool_call_id

                if self.logger is not None:
                    self.logger.log(
                        f"tool_call_received id={turn_id} tool_call_id={tool_call_id} name={name} "
                        f"status={activity.get('status')} args_len={len(arguments_raw)}",
                        source="app",
                    )
                    if activity.get("status") == "executed":
                        result = str(activity.get("result") or "")
                        self.logger.log(
                            f"tool_call_execute_success id={turn_id} tool_call_id={tool_call_id} "
                            f"name={name} output_len={len(result)}",
                            source="app",
                        )
                    elif activity.get("status") not in {"dry_run", "denied_by_policy"}:
                        self.logger.log(
                            f"tool_call_execute_error id={turn_id} tool_call_id={tool_call_id} "
                            f"name={name} error={activity.get('error')}",
                            source="app",
                        )

                tool_activity.append(activity)

                if tool_runtime.mode != "dry_run":
                    result_text = str(activity.get("result") or activity.get("error") or "")
                    working_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": result_text,
                        }
                    )
                    if self.logger is not None:
                        self.logger.log(
                            f"tool_call_result_injected id={turn_id} tool_call_id={tool_call_id} "
                            f"name={name} content_len={len(result_text)}",
                            source="app",
                        )

            if tool_runtime.mode == "dry_run":
                final_answer_text = "".join(answer_parts)
                final_think_text = "".join(think_parts)
                break

        ended = time.time()
        elapsed = max(0.0, ended - started)
        throughput = None
        token_counts = None
        if aggregate_prompt_tokens or aggregate_completion_tokens or aggregate_total_tokens:
            if aggregate_total_tokens <= 0:
                aggregate_total_tokens = aggregate_prompt_tokens + aggregate_completion_tokens
            token_counts = {
                "prompt_tokens": aggregate_prompt_tokens,
                "completion_tokens": aggregate_completion_tokens,
                "total_tokens": aggregate_total_tokens,
            }
        if token_counts is not None and token_counts.get("completion_tokens", 0) > 0 and elapsed > 0:
            throughput = {"tokens_per_s": token_counts["completion_tokens"] / elapsed}
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
        if all_tool_calls_for_gen:
            gen["tool_calls"] = all_tool_calls_for_gen
        if final_finish_reason is not None:
            gen["finish_reason"] = final_finish_reason
        context_report = build_retry_report(
            messages,
            working_messages,
            strategy="overflow_retry",
            context_window=context_window,
            reserved_generation_tokens=reserved_generation,
            overflow_retries=overflow_retries,
            fit=True,
            prompt_tokens=final_context_prompt_tokens,
            dropped_roles=dropped_roles,
        ).to_dict()
        trimmed_messages = None
        if tool_runtime.mode == "dry_run" and tool_activity:
            trimmed_messages = list(working_messages)
        else:
            trimmed_messages = append_assistant_history(
                list(working_messages),
                think=final_think_text,
                answer=final_answer_text,
                strip_think=bool(self.args.history_strip_think),
            )
        record = TurnRecord(
            raw="".join(aggregate_raw_parts),
            think="".join(aggregate_think_parts),
            answer="".join(aggregate_answer_parts),
            ended_in_think=ended_in_think,
            backend=self.backend_name,
            model_id=self.resolved_model_id,
            gen=gen,
            timing={"start": started, "end": ended, "elapsed": elapsed},
            token_counts=token_counts,
            throughput=throughput,
            knobs=final_knob_report,
            context=context_report,
            tool_activity=tool_activity or None,
            trimmed_messages=trimmed_messages,
        )
        emit(Finish(turn_id=turn_id, record=record))
        if self.logger is not None:
            self.logger.log(
                f"turn_finish id={turn_id} elapsed_s={record.timing.get('elapsed', 0):.3f} "
                f"think_chars={len(record.think)} answer_chars={len(record.answer)}",
                source="transport",
            )
