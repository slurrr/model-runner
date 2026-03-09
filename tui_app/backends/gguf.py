from __future__ import annotations

import argparse
import json
import os
import re
import time

from tui_app.backends.base import EventEmitter
from tui_app.events import AnswerDelta, Error, Finish, Meta, ThinkDelta, TurnRecord, TurnStart
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

    def __init__(self, llm, args: argparse.Namespace, resolved_model_id: str, logger: FileLogger | None = None):
        self.llm = llm
        self.args = args
        self.resolved_model_id = resolved_model_id
        self.logger = logger

    def close(self) -> None:
        if self.logger is not None:
            self.logger.close()

    def get_recent_logs(self, n: int = 80) -> list[str]:
        if self.logger is None:
            return []
        return self.logger.get_recent_logs(n=n)

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
                f"max_new_tokens={self.args.max_new_tokens}"
            )
        router = ThinkRouter(assume_think=self.args.assume_think)
        working_messages = list(messages)

        raw_parts: list[str] = []
        think_parts: list[str] = []
        answer_parts: list[str] = []
        think_counter_text = ""
        think_counter_tokens = 0

        max_tokens = self.args.max_new_tokens
        effective_max_tokens = max_tokens
        stop = self.args.stop_strings or None
        sampling_kwargs = _common_gguf_sampling_kwargs(self.args, stop=stop)

        def _is_context_overflow(exc: Exception) -> bool:
            text = str(exc).lower()
            return "context window" in text or ("requested tokens" in text and "exceed" in text)

        def _drop_oldest_turn() -> bool:
            start_idx = 1 if working_messages and working_messages[0].get("role") == "system" else 0
            for idx in range(start_idx, len(working_messages)):
                role = working_messages[idx].get("role")
                if role in {"user", "assistant", "tool"}:
                    working_messages.pop(idx)
                    return True
            return False

        def _shrink_max_tokens() -> bool:
            nonlocal effective_max_tokens
            if effective_max_tokens <= 64:
                return False
            effective_max_tokens = max(64, int(effective_max_tokens * 0.75))
            return effective_max_tokens > 0

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

        def _sampling_kwargs_for_retry():
            kwargs = dict(sampling_kwargs)
            kwargs["max_tokens"] = effective_max_tokens
            return kwargs

        if self.args.prompt_mode == "plain":
            while True:
                try:
                    prompt = self._fallback_plain_prompt(working_messages)
                    response = self.llm.create_completion(
                        prompt=prompt,
                        **{
                            **_sampling_kwargs_for_retry(),
                            "stop": stop or ["\nUser:", "\nSystem:"],
                        },
                    )
                    text = response["choices"][0].get("text", "")
                    raw_parts.append(text)
                    for channel, part in router.feed(text):
                        if channel == "think":
                            _emit_think(part)
                        else:
                            answer_parts.append(part)
                            emit(AnswerDelta(turn_id=turn_id, text=part))
                    break
                except Exception as exc:
                    if _is_context_overflow(exc) and (_drop_oldest_turn() or _shrink_max_tokens()):
                        continue
                    if self.logger is not None:
                        self.logger.log(f"turn_error id={turn_id} error={exc}")
                    emit(Error(turn_id=turn_id, message=f"Generation failed in plain mode: {exc}"))
                    return
        else:
            while True:
                try:
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

                        raw_parts.append(delta)
                        for channel, text in router.feed(delta):
                            if channel == "think":
                                _emit_think(text)
                            else:
                                answer_parts.append(text)
                                emit(AnswerDelta(turn_id=turn_id, text=text))
                    break
                except Exception as exc:
                    if _is_context_overflow(exc) and (_drop_oldest_turn() or _shrink_max_tokens()):
                        continue
                    try:
                        prompt = self._fallback_plain_prompt(working_messages)
                        response = self.llm.create_completion(
                            prompt=prompt,
                            **{
                                **_sampling_kwargs_for_retry(),
                                "stop": stop or ["\nUser:", "\nSystem:"],
                            },
                        )
                        text = response["choices"][0].get("text", "")
                        raw_parts.append(text)
                        for channel, part in router.feed(text):
                            if channel == "think":
                                _emit_think(part)
                            else:
                                answer_parts.append(part)
                                emit(AnswerDelta(turn_id=turn_id, text=part))
                        break
                    except Exception as fallback_exc:
                        if _is_context_overflow(fallback_exc) and (_drop_oldest_turn() or _shrink_max_tokens()):
                            continue
                        if self.logger is not None:
                            self.logger.log(f"turn_error id={turn_id} error={exc}; fallback={fallback_exc}")
                        emit(Error(turn_id=turn_id, message=f"Generation failed: {exc}; fallback failed: {fallback_exc}"))
                        return

        for channel, text in router.flush():
            if channel == "think":
                _emit_think(text)
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
            timing={"start": started, "end": ended, "elapsed": max(0.0, ended - started)},
            trimmed_messages=working_messages,
        )
        emit(Finish(turn_id=turn_id, record=record))
        if self.logger is not None:
            self.logger.log(
                f"turn_finish id={turn_id} elapsed_s={record.timing.get('elapsed', 0):.3f} "
                f"think_chars={len(record.think)} answer_chars={len(record.answer)}"
            )


def create_session(args: argparse.Namespace) -> GGUFSession:
    model_path = normalize_model_path(args.model_id)
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {model_path}")
    if not model_path.lower().endswith(".gguf"):
        raise RuntimeError(f"Expected a .gguf file, got: {model_path}")

    logger = FileLogger.from_value(getattr(args, "gguf_log_file", ""), "gguf")
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
        logger.log(f"session_init model={model_path} n_ctx={args.n_ctx} n_gpu_layers={args.n_gpu_layers}")
    llm = Llama(
        model_path=model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=False,
    )
    template_kind, template_value = resolve_gguf_chat_template_spec(args, config_path=args._config_path)
    template_status = "auto"
    if args.prompt_mode == "plain":
        template_status = "auto (plain mode)"
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
        elif template_kind == "format":
            llm.chat_handler = None
            llm.chat_format = template_value
            template_status = f"format:{template_value}"
    print(f"GGUF chat template: {template_status}")
    if logger is not None:
        logger.log(f"chat_template={template_status}")
    return GGUFSession(llm=llm, args=args, resolved_model_id=model_path, logger=logger)
