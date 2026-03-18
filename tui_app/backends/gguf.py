from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Callable

from tui_app.backends.base import EventEmitter
from tui_app.context_policy import (
    build_context_limit_error,
    build_retry_report,
    drop_oldest_history_message,
    is_context_overflow_text,
    reserve_generation_tokens,
    trim_messages_to_budget,
)
from tui_app.events import AnswerDelta, Error, Finish, Meta, ThinkDelta, TurnRecord, TurnStart
from tui_app.history_control import append_assistant_history
from tui_app.knobs import SUPPORTED_KNOBS, finalize_knob_report, unsupported_user_set
from tui_app.log_file import FileLogger
from tui_app.think_router import ThinkRouter


def normalize_model_path(raw_path: str) -> str:
    path = os.path.expanduser(raw_path.strip())
    win_drive = re.match(r"^([A-Za-z]):[\\/](.*)$", path)
    if win_drive:
        drive = win_drive.group(1).lower()
        rest = win_drive.group(2).replace("\\", "/")
        path = f"/mnt/{drive}/{rest}"
    return os.path.abspath(path)


def resolve_path_maybe_relative(path: str, config_path: str | None = None) -> str:
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    if config_path:
        cfg_dir = os.path.dirname(config_path)
        candidate = os.path.abspath(os.path.join(cfg_dir, expanded))
        if os.path.exists(candidate):
            return candidate
    return os.path.abspath(expanded)


def resolve_gguf_chat_template_spec(args: argparse.Namespace, config_path: str | None) -> tuple[str, str]:
    spec = (args.chat_template or "").strip()
    if not spec:
        return "auto", ""
    path = resolve_path_maybe_relative(spec, config_path=config_path)
    if os.path.isfile(path):
        return "file", path
    return "format", spec


def load_chat_template_text(path: str) -> str:
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise RuntimeError(f"Template JSON must be an object: {path}")
        template = data.get("chat_template") or data.get("template")
        if not isinstance(template, str) or not template.strip():
            raise RuntimeError(f"No 'chat_template' or 'template' found in: {path}")
        return template

    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if not text.strip():
        raise RuntimeError(f"Template file is empty: {path}")
    return text


def _common_gguf_sampling_kwargs(args: argparse.Namespace, stop):
    kwargs = {
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stop": stop,
    }
    if args.top_k is not None:
        kwargs["top_k"] = args.top_k
    if args.min_p is not None:
        kwargs["min_p"] = args.min_p
    if args.typical_p is not None:
        kwargs["typical_p"] = args.typical_p
    if args.repetition_penalty not in (None, 1.0):
        kwargs["repeat_penalty"] = args.repetition_penalty
    return {k: v for k, v in kwargs.items() if v is not None}


class GGUFSession:
    backend_name = "gguf"

    def __init__(
        self,
        llm,
        args: argparse.Namespace,
        resolved_model_id: str,
        logger: FileLogger | None = None,
        prompt_token_counter: Callable[[list[dict[str, str]]], int | None] | None = None,
        template_info: dict[str, object] | None = None,
    ):
        self.llm = llm
        self.args = args
        self.resolved_model_id = resolved_model_id
        self.logger = logger
        self.prompt_token_counter = prompt_token_counter
        self.template_info = dict(template_info or {})

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

    def describe(self) -> dict[str, object]:
        return {
            "template_control_level": "local_template",
            **self.template_info,
        }

    def _fallback_plain_prompt(self, messages: list[dict[str, str]]) -> str:
        system = self.args.system or "You are a helpful assistant."
        parts = [f"System: {system}"]
        for msg in messages:
            role = msg.get("role", "message").capitalize()
            parts.append(f"{role}: {msg.get('content', '')}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def generate_turn(self, turn_id: int, messages: list[dict[str, str]], emit: EventEmitter) -> None:
        emit(TurnStart(turn_id=turn_id))
        started = time.time()
        if self.logger is not None:
            self.logger.log(
                f"turn_start id={turn_id} messages={len(messages)} prompt_mode={self.args.prompt_mode} "
                f"max_new_tokens={self.args.max_new_tokens}",
                source="backend",
            )
        router = ThinkRouter(assume_think=self.args.assume_think)
        working_messages = list(messages)

        raw_parts: list[str] = []
        think_parts: list[str] = []
        answer_parts: list[str] = []
        think_counter_text = ""
        think_counter_tokens = 0
        generated_counter_text = ""
        generated_counter_tokens = 0
        prompt_tokens: int | None = None
        knob_notes: list[str] = []
        context_report: dict[str, object] | None = None
        overflow_retries = 0
        dropped_roles: list[str] = []

        max_tokens = self.args.max_new_tokens
        stop = self.args.stop_strings or None
        sampling_kwargs = _common_gguf_sampling_kwargs(self.args, stop=stop)
        reserved_generation = reserve_generation_tokens(int(self.args.n_ctx), max_tokens)

        def _is_context_overflow(exc: Exception) -> bool:
            return is_context_overflow_text(str(exc))

        def _drop_oldest_turn() -> bool:
            dropped_chunk = drop_oldest_history_message(working_messages)
            if dropped_chunk is None:
                return False
            dropped_roles.extend(dropped_chunk)
            return True

        def _count_tokens(text: str) -> int:
            if not text:
                return 0
            data = text.encode("utf-8", "ignore")
            try:
                return len(self.llm.tokenize(data, add_bos=False, special=True))
            except TypeError:
                try:
                    return len(self.llm.tokenize(data, add_bos=False))
                except TypeError:
                    return len(self.llm.tokenize(data))

        def _emit_think(text: str) -> None:
            nonlocal think_counter_text, think_counter_tokens
            if not text:
                return
            think_parts.append(text)
            emit(ThinkDelta(turn_id=turn_id, text=text))
            # Near-exact tokenizer-based count of emitted think text.
            think_counter_text += text
            current = _count_tokens(think_counter_text)
            token_inc = current - think_counter_tokens
            think_counter_tokens = current
            if token_inc > 0:
                emit(Meta(turn_id=turn_id, key="think_tokens_inc", value=token_inc))

        def _emit_generated(text: str) -> None:
            nonlocal generated_counter_text, generated_counter_tokens
            if not text:
                return
            generated_counter_text += text
            current = _count_tokens(generated_counter_text)
            token_inc = current - generated_counter_tokens
            generated_counter_tokens = current
            if token_inc > 0:
                emit(Meta(turn_id=turn_id, key="generated_tokens_inc", value=token_inc))

        def _sampling_kwargs_for_retry():
            kwargs = dict(sampling_kwargs)
            kwargs["max_tokens"] = max_tokens
            return kwargs

        if self.prompt_token_counter is not None:
            def _measure(trimmed_messages):
                count = self.prompt_token_counter(trimmed_messages)
                if count is None:
                    raise RuntimeError("GGUF prompt token counter returned no value.")
                return int(count), None

            trimmed_messages, prompt_tokens, _unused, report = trim_messages_to_budget(
                working_messages,
                measure_fn=_measure,
                context_window=int(self.args.n_ctx),
                reserved_generation_tokens=reserved_generation,
                strategy="exact_preflight",
            )
            working_messages = list(trimmed_messages)
            if not report.fit:
                detail = build_context_limit_error(report)
                if self.logger is not None:
                    self.logger.log(f"turn_error id={turn_id} error={detail}", source="backend")
                emit(Error(turn_id=turn_id, message=detail))
                return
            context_report = report.to_dict()

        if self.args.prompt_mode == "plain":
            while True:
                try:
                    used_mode = "plain"
                    prompt = self._fallback_plain_prompt(working_messages)
                    response = self.llm.create_completion(
                        prompt=prompt,
                        **{
                            **_sampling_kwargs_for_retry(),
                            "stop": stop or ["\nUser:", "\nSystem:"],
                        },
                    )
                    text = response["choices"][0].get("text", "")
                    _emit_generated(text)
                    raw_parts.append(text)
                    for channel, part in router.feed(text):
                        if channel == "think":
                            _emit_think(part)
                        else:
                            answer_parts.append(part)
                            emit(AnswerDelta(turn_id=turn_id, text=part))
                    break
                except Exception as exc:
                    if _is_context_overflow(exc):
                        if _drop_oldest_turn():
                            overflow_retries += 1
                            continue
                        detail = build_context_limit_error(
                            build_retry_report(
                                messages,
                                working_messages,
                                strategy="overflow_retry",
                                context_window=int(self.args.n_ctx),
                                reserved_generation_tokens=reserved_generation,
                                overflow_retries=overflow_retries,
                                fit=False,
                                prompt_tokens=prompt_tokens,
                                dropped_roles=dropped_roles,
                            )
                        )
                        if self.logger is not None:
                            self.logger.log(f"turn_error id={turn_id} error={detail}", source="backend")
                        emit(Error(turn_id=turn_id, message=detail))
                        return
                    if self.logger is not None:
                        self.logger.log(f"turn_error id={turn_id} error={exc}", source="backend")
                    emit(Error(turn_id=turn_id, message=f"Generation failed in plain mode: {exc}"))
                    return
        else:
            while True:
                try:
                    used_mode = "chat"
                    stream = self.llm.create_chat_completion(
                        messages=working_messages,
                        stream=True,
                        **_sampling_kwargs_for_retry(),
                    )

                    for chunk in stream:
                        delta = ""
                        choices = chunk.get("choices", [])
                        if choices:
                            delta_obj = choices[0].get("delta", {}) or {}
                            delta = delta_obj.get("content", "") or ""
                            if not delta:
                                msg = choices[0].get("message", {}) or {}
                                delta = msg.get("content", "") or ""
                        if not delta:
                            continue

                        _emit_generated(delta)
                        raw_parts.append(delta)
                        for channel, text in router.feed(delta):
                            if channel == "think":
                                _emit_think(text)
                            else:
                                answer_parts.append(text)
                                emit(AnswerDelta(turn_id=turn_id, text=text))
                    break
                except Exception as exc:
                    if _is_context_overflow(exc):
                        if _drop_oldest_turn():
                            overflow_retries += 1
                            continue
                        detail = build_context_limit_error(
                            build_retry_report(
                                messages,
                                working_messages,
                                strategy="overflow_retry",
                                context_window=int(self.args.n_ctx),
                                reserved_generation_tokens=reserved_generation,
                                overflow_retries=overflow_retries,
                                fit=False,
                                prompt_tokens=prompt_tokens,
                                dropped_roles=dropped_roles,
                            )
                        )
                        if self.logger is not None:
                            self.logger.log(f"turn_error id={turn_id} error={detail}", source="backend")
                        emit(Error(turn_id=turn_id, message=detail))
                        return
                    try:
                        used_mode = "plain"
                        knob_notes.append("GGUF: chat API failed; fell back to plain completion")
                        prompt = self._fallback_plain_prompt(working_messages)
                        response = self.llm.create_completion(
                            prompt=prompt,
                            **{
                                **_sampling_kwargs_for_retry(),
                                "stop": stop or ["\nUser:", "\nSystem:"],
                            },
                        )
                        text = response["choices"][0].get("text", "")
                        _emit_generated(text)
                        raw_parts.append(text)
                        for channel, part in router.feed(text):
                            if channel == "think":
                                _emit_think(part)
                            else:
                                answer_parts.append(part)
                                emit(AnswerDelta(turn_id=turn_id, text=part))
                        break
                    except Exception as fallback_exc:
                        if _is_context_overflow(fallback_exc):
                            if _drop_oldest_turn():
                                overflow_retries += 1
                                continue
                            detail = build_context_limit_error(
                                build_retry_report(
                                    messages,
                                    working_messages,
                                    strategy="overflow_retry",
                                    context_window=int(self.args.n_ctx),
                                    reserved_generation_tokens=reserved_generation,
                                    overflow_retries=overflow_retries,
                                    fit=False,
                                    prompt_tokens=prompt_tokens,
                                    dropped_roles=dropped_roles,
                                )
                            )
                            if self.logger is not None:
                                self.logger.log(f"turn_error id={turn_id} error={detail}", source="backend")
                            emit(Error(turn_id=turn_id, message=detail))
                            return
                        if self.logger is not None:
                            self.logger.log(f"turn_error id={turn_id} error={exc}; fallback={fallback_exc}", source="backend")
                        emit(Error(turn_id=turn_id, message=f"Generation failed: {exc}; fallback failed: {fallback_exc}"))
                        return

        for channel, text in router.flush():
            if channel == "think":
                _emit_think(text)
            else:
                answer_parts.append(text)
                emit(AnswerDelta(turn_id=turn_id, text=text))

        ended = time.time()
        completion_tokens = _count_tokens("".join(raw_parts))
        if self.prompt_token_counter is not None:
            try:
                prompt_tokens = self.prompt_token_counter(working_messages)
            except Exception:
                prompt_tokens = None
        if prompt_tokens is not None:
            emit(Meta(turn_id=turn_id, key="prompt_tokens", value=int(prompt_tokens)))
        emit(Meta(turn_id=turn_id, key="completion_tokens", value=int(completion_tokens)))
        total_tokens = None
        if prompt_tokens is not None:
            total_tokens = int(prompt_tokens) + int(completion_tokens)
            emit(Meta(turn_id=turn_id, key="total_tokens", value=total_tokens))
        elapsed = max(0.0, ended - started)
        throughput = None
        if completion_tokens > 0 and elapsed > 0:
            throughput = {"tokens_per_s": completion_tokens / elapsed}
        if context_report is None:
            context_report = build_retry_report(
                messages,
                working_messages,
                strategy="overflow_retry",
                context_window=int(self.args.n_ctx),
                reserved_generation_tokens=reserved_generation,
                overflow_retries=overflow_retries,
                fit=True,
                prompt_tokens=prompt_tokens,
                dropped_roles=dropped_roles,
            ).to_dict()
        token_counts = {"completion_tokens": int(completion_tokens)}
        if prompt_tokens is not None:
            token_counts["prompt_tokens"] = int(prompt_tokens)
        if total_tokens is not None:
            token_counts["total_tokens"] = int(total_tokens)
        knob_sent = {
            "max_new_tokens": int(max_tokens),
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
        }
        if self.args.top_k is not None:
            knob_sent["top_k"] = self.args.top_k
        if self.args.min_p is not None:
            knob_sent["min_p"] = self.args.min_p
        if self.args.typical_p is not None:
            knob_sent["typical_p"] = self.args.typical_p
        if self.args.repetition_penalty not in (None, 1.0):
            knob_sent["repetition_penalty"] = self.args.repetition_penalty
        if stop is not None:
            knob_sent["stop_strings"] = list(stop) if isinstance(stop, list) else stop
        elif used_mode == "plain":
            knob_sent["stop_strings"] = ["\nUser:", "\nSystem:"]
        knob_report = finalize_knob_report(
            sent=knob_sent,
            supported=SUPPORTED_KNOBS["gguf"],
            ignored=unsupported_user_set(self.args, "gguf"),
            notes=knob_notes,
        )
        answer_text = "".join(answer_parts)
        record = TurnRecord(
            raw="".join(raw_parts),
            think="".join(think_parts),
            answer=answer_text,
            ended_in_think=(router.mode == "think"),
            backend=self.backend_name,
            model_id=self.resolved_model_id,
            gen={
                "max_new_tokens": max_tokens,
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
                "top_k": self.args.top_k,
                "min_p": self.args.min_p,
                "typical_p": self.args.typical_p,
                "repetition_penalty": self.args.repetition_penalty,
                "prompt_mode": self.args.prompt_mode,
                "chat_template": self.args.chat_template,
            },
            timing={"start": started, "end": ended, "elapsed": elapsed},
            token_counts=token_counts,
            throughput=throughput,
            knobs=knob_report,
            context=context_report,
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


def create_session(args: argparse.Namespace) -> GGUFSession:
    model_path = normalize_model_path(args.model_id)
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {model_path}")
    if not model_path.lower().endswith(".gguf"):
        raise RuntimeError(f"Expected a .gguf file, got: {model_path}")

    logger = FileLogger.from_value(
        getattr(args, "gguf_log_file", ""),
        "backend",
        config_path=getattr(args, "_config_path", None),
    )
    try:
        from llama_cpp import Llama, _internals

        original_close = _internals.LlamaModel.close

        def safe_close(self):
            if not hasattr(self, "sampler"):
                self.sampler = None
            if not hasattr(self, "custom_samplers"):
                self.custom_samplers = []
            if not hasattr(self, "_exit_stack"):
                return
            try:
                original_close(self)
            except Exception:
                return

        _internals.LlamaModel.close = safe_close
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: llama-cpp-python. Install with: .venv/bin/pip install llama-cpp-python"
        ) from exc

    print(f"Loading GGUF model: {model_path}")
    if logger is not None:
        logger.log(f"session_init model={model_path} n_ctx={args.n_ctx} n_gpu_layers={args.n_gpu_layers}", source="app")
    llm = Llama(
        model_path=model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=False,
    )
    template_kind, template_value = resolve_gguf_chat_template_spec(args, config_path=args._config_path)
    template_status = "auto"
    requested_template = (args.chat_template or "").strip()
    template_requested_value = requested_template
    if requested_template:
        try:
            candidate = resolve_path_maybe_relative(requested_template, config_path=args._config_path)
            if os.path.isfile(candidate):
                template_requested_value = candidate
        except Exception:
            pass
    prompt_token_counter: Callable[[list[dict[str, str]]], int | None] | None = None

    def _tokenize_prompt_text(prompt_text: str) -> int:
        data = prompt_text.encode("utf-8", "ignore")
        try:
            return len(llm.tokenize(data, add_bos=False, special=True))
        except TypeError:
            try:
                return len(llm.tokenize(data, add_bos=False))
            except TypeError:
                return len(llm.tokenize(data))

    def _fallback_plain_prompt(messages: list[dict[str, str]]) -> str:
        system = args.system or "You are a helpful assistant."
        parts = [f"System: {system}"]
        for msg in messages:
            role = msg.get("role", "message").capitalize()
            parts.append(f"{role}: {msg.get('content', '')}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _plain_prompt_tokens(messages: list[dict[str, str]]) -> int:
        prompt = _fallback_plain_prompt(messages)
        return _tokenize_prompt_text(prompt)

    if args.prompt_mode == "plain":
        template_status = "auto (plain mode)"
        prompt_token_counter = _plain_prompt_tokens
    else:
        if template_kind == "file":
            try:
                from llama_cpp.llama_chat_format import Jinja2ChatFormatter, chat_formatter_to_chat_completion_handler
            except Exception as exc:
                raise RuntimeError(
                    "This llama-cpp-python build does not support Jinja chat format helpers "
                    "(Jinja2ChatFormatter)."
                ) from exc

            template_text = load_chat_template_text(template_value)
            try:
                bos = llm.detokenize([llm.token_bos()], special=True).decode("utf-8", "ignore")
            except Exception:
                bos = ""
            try:
                eos = llm.detokenize([llm.token_eos()], special=True).decode("utf-8", "ignore")
            except Exception:
                eos = ""
            formatter = Jinja2ChatFormatter(
                template=template_text,
                bos_token=bos or "",
                eos_token=eos or "",
                add_generation_prompt=True,
            )
            llm.chat_handler = chat_formatter_to_chat_completion_handler(formatter)
            llm.chat_format = None
            template_status = f"file:{template_value}"

            def _file_template_prompt_tokens(messages: list[dict[str, str]]) -> int:
                rendered = formatter(messages=messages)
                return _tokenize_prompt_text(rendered.prompt)

            prompt_token_counter = _file_template_prompt_tokens
        elif template_kind == "format":
            llm.chat_handler = None
            llm.chat_format = template_value
            template_status = f"format:{template_value}"
    print(f"GGUF chat template: {template_status}")
    if logger is not None:
        logger.log(f"chat_template={template_status}", source="app")
    template_info = {
        "chat_template_requested": template_requested_value,
        "chat_template_applied": bool(requested_template and args.prompt_mode != "plain"),
        "chat_template_reason": (
            "applied" if requested_template and args.prompt_mode != "plain" else ("ignored_prompt_mode" if requested_template else "empty_default")
        ),
    }
    return GGUFSession(
        llm=llm,
        args=args,
        resolved_model_id=model_path,
        logger=logger,
        prompt_token_counter=prompt_token_counter,
        template_info=template_info,
    )
