from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass

import torch
from jinja2 import Environment

from tui_app.backends.base import EventEmitter
from tui_app.context_policy import build_context_limit_error, trim_messages_to_budget
from tui_app.events import AnswerDelta, Error, Finish, Meta, ThinkDelta, TurnRecord, TurnStart
from tui_app.history_control import append_assistant_history
from tui_app.knobs import SUPPORTED_KNOBS, finalize_knob_report, unsupported_user_set
from tui_app.log_file import FileLogger
from tui_app.think_router import ThinkRouter


def normalize_model_dir(raw: str) -> str:
    value = os.path.expanduser((raw or "").strip())
    win_drive = re.match(r"^([A-Za-z]):[\\/](.*)$", value)
    if win_drive:
        drive = win_drive.group(1).lower()
        rest = win_drive.group(2).replace("\\", "/")
        value = f"/mnt/{drive}/{rest}"
    return os.path.abspath(value)


def resolve_exl2_model_dir(
    model_ref: str,
    *,
    model_path: str | None = None,
    exl2_repo_path: str | None = None,
    config_path: str | None = None,
) -> str:
    primary = normalize_model_dir(model_ref)
    if os.path.isdir(primary):
        return primary

    if os.path.isabs(os.path.expanduser((model_ref or "").strip())):
        return primary

    candidates: list[str] = []
    if model_path:
        candidates.append(os.path.join(resolve_path_maybe_relative(model_path, config_path=config_path), model_ref))
    if exl2_repo_path:
        candidates.append(os.path.join(resolve_path_maybe_relative(exl2_repo_path, config_path=config_path), model_ref))

    for candidate in candidates:
        normalized = normalize_model_dir(candidate)
        if os.path.isdir(normalized):
            return normalized
    return primary


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


def _read_json_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Template JSON must be an object.")
    template = data.get("chat_template") or data.get("template")
    if not template:
        raise ValueError(f"No 'chat_template' or 'template' field in {path}")
    if not isinstance(template, str):
        raise ValueError("Template must be a string.")
    return template


def resolve_chat_template(template_spec: str, model_dir: str, config_path: str | None = None) -> str | None:
    """
    Returns a Jinja chat template text or None for fallback.

    Rules:
    - empty / 'default': try tokenizer_config.json chat_template; else None
    - 'search': same as default for EXL2 (tokenizer_config.json only)
    - file path: read file (or parse JSON)
    - otherwise: treat as raw template text
    """
    spec = (template_spec or "").strip()
    lowered = spec.lower()

    if not spec or lowered in {"default", "tokenizer_config", "search", "tokenizer_config_search"}:
        candidate = os.path.join(model_dir, "tokenizer_config.json")
        if os.path.isfile(candidate):
            try:
                return _read_json_template(candidate)
            except Exception:
                return None
        return None

    path = resolve_path_maybe_relative(spec, config_path=config_path)
    if os.path.isfile(path):
        if path.endswith(".json"):
            return _read_json_template(path)
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()

    # Treat as inline template text.
    return spec


DEFAULT_TEMPLATE = """\
{%- set ns = namespace(system_prompt=\"\") -%}
{%- for message in messages -%}
  {%- if message[\"role\"] == \"system\" -%}
    {%- set ns.system_prompt = message[\"content\"] -%}
  {%- endif -%}
{%- endfor -%}
{{- bos_token -}}
{%- if ns.system_prompt -%}
System: {{ ns.system_prompt }}
{%- endif -%}
{%- for message in messages -%}
  {%- if message[\"role\"] == \"user\" -%}
User: {{ message[\"content\"] }}
  {%- elif message[\"role\"] == \"assistant\" -%}
Assistant: {{ message[\"content\"] }}
  {%- endif -%}
{%- endfor -%}
Assistant:
"""


def render_chat_template(template_text: str, *, messages: list[dict[str, str]], bos_token: str, eos_token: str) -> str:
    env = Environment(autoescape=False)
    tmpl = env.from_string(template_text)
    return tmpl.render(
        messages=messages,
        bos_token=bos_token or "",
        eos_token=eos_token or "",
        add_generation_prompt=True,
    )


def _count_tokens(token_obj) -> int:
    if token_obj is None:
        return 0
    if isinstance(token_obj, int):
        return 1
    shape = getattr(token_obj, "shape", None)
    if isinstance(shape, (tuple, list)) and shape:
        try:
            return int(shape[-1])
        except Exception:
            pass
    if hasattr(token_obj, "numel"):
        try:
            return int(token_obj.numel())
        except Exception:
            pass
    if isinstance(token_obj, (list, tuple)):
        if not token_obj:
            return 0
        first = token_obj[0]
        if isinstance(first, (list, tuple)):
            return int(len(first))
        return int(len(token_obj))
    return 0


def _iter_token_ids(token_obj) -> list[int]:
    if token_obj is None:
        return []
    if isinstance(token_obj, int):
        return [int(token_obj)]
    if hasattr(token_obj, "view"):
        try:
            flat = token_obj.view(-1).tolist()
            return [int(x) for x in flat]
        except Exception:
            pass
    if isinstance(token_obj, (list, tuple)):
        out: list[int] = []
        for item in token_obj:
            if isinstance(item, (list, tuple)):
                out.extend(int(x) for x in item)
            else:
                try:
                    out.append(int(item))
                except Exception:
                    continue
        return out
    return []


def _sanitize_messages(messages: list[dict[str, str]] | list[object]) -> list[dict[str, str]]:
    clean: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "") or "").strip()
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        content = item.get("content", "")
        if content is None:
            content = ""
        clean.append({"role": role, "content": str(content)})
    return clean


def _strip_trailing_assistant_if_matches(
    messages: list[dict[str, str]],
    assistant_text: str,
) -> list[dict[str, str]]:
    if not assistant_text:
        return messages
    if not messages:
        return messages
    last = messages[-1]
    if last.get("role") == "assistant" and (last.get("content") or "") == assistant_text:
        return messages[:-1]
    return messages


def _build_stop_conditions(tokenizer, args: argparse.Namespace) -> list[object]:
    conditions: list[object] = []
    seen: set[object] = set()

    def add(value: object) -> None:
        if value is None:
            return
        if isinstance(value, int) and value < 0:
            return
        if isinstance(value, str) and not value:
            return
        if value in seen:
            return
        seen.add(value)
        conditions.append(value)

    # Always include EOS when exposed by tokenizer.
    add(getattr(tokenizer, "eos_token_id", None))

    # Optional model/backend-provided token markers.
    configured_markers = list(getattr(args, "exl2_stop_tokens", None) or [])
    if not configured_markers:
        configured_markers = ["<end_of_turn>", "<start_of_turn>"]

    single_id = getattr(tokenizer, "single_id", None)
    if callable(single_id):
        for marker in configured_markers:
            try:
                add(single_id(marker))
            except Exception:
                continue

    # String stops still supported for parity with other backends.
    for s in (args.stop_strings or []):
        add(s)
    return conditions


def apply_context_limit_exl2(
    tokenizer,
    template_text: str,
    messages: list[dict[str, str]],
    *,
    max_seq_len: int,
    min_free_tokens: int,
) -> tuple[torch.Tensor, list[dict[str, str]], str, dict[str, object]]:
    sanitized = _sanitize_messages(messages)

    def _measure(trimmed):
        prompt_text = render_chat_template(
            template_text,
            messages=trimmed,
            bos_token=getattr(tokenizer, "bos_token", ""),
            eos_token=getattr(tokenizer, "eos_token", ""),
        )
        input_ids = tokenizer.encode(prompt_text, add_bos=False, add_eos=False, encode_special_tokens=True)
        length = _count_tokens(input_ids)
        if length <= 0:
            raise RuntimeError("Tokenizer returned no input ids for rendered EXL2 prompt.")
        return length, (input_ids, prompt_text)

    trimmed, _length, payload, report = trim_messages_to_budget(
        sanitized,
        measure_fn=_measure,
        context_window=max_seq_len,
        reserved_generation_tokens=min_free_tokens,
        strategy="exact_preflight",
    )
    if not report.fit:
        raise RuntimeError(build_context_limit_error(report))
    input_ids, prompt_text = payload
    return input_ids, trimmed, prompt_text, report.to_dict()


def _try_import_exllamav2(exl2_repo_path: str | None):
    try:
        import exllamav2  # noqa: F401
        return
    except Exception:
        pass

    candidates = []
    if exl2_repo_path:
        candidates.append(exl2_repo_path)
    candidates.append(os.path.expanduser("~/ml/exllamav2"))

    for candidate in candidates:
        if not candidate:
            continue
        path = os.path.abspath(os.path.expanduser(candidate))
        if not os.path.isdir(path):
            continue
        if path not in sys.path:
            sys.path.insert(0, path)
        try:
            import exllamav2  # noqa: F401
            return
        except Exception:
            continue

    # Import one last time to surface the true error.
    import exllamav2  # type: ignore # noqa: F401


@dataclass
class _Exl2Runtime:
    model: object
    cache: object
    tokenizer: object
    generator: object
    backend_info: dict[str, object]


class EXL2Session:
    backend_name = "exl2"

    def __init__(
        self,
        runtime: _Exl2Runtime,
        args: argparse.Namespace,
        resolved_model_id: str,
        template_info: dict[str, object] | None = None,
        logger: FileLogger | None = None,
    ):
        self.runtime = runtime
        self.args = args
        self.resolved_model_id = resolved_model_id
        self.template_info = dict(template_info or {})
        self.logger = logger

    def describe(self) -> dict[str, object]:
        return {
            "template_control_level": "local_template",
            **dict(self.runtime.backend_info),
            **self.template_info,
        }

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

    def _build_settings(self):
        from exllamav2.generator import ExLlamaV2Sampler

        settings = ExLlamaV2Sampler.Settings(
            temperature=self.args.temperature,
            top_k=self.args.top_k or 0,
            top_p=self.args.top_p,
            min_p=self.args.min_p or 0,
            typical=self.args.typical_p or 0,
            token_repetition_penalty=self.args.repetition_penalty,
            token_frequency_penalty=(self.args.frequency_penalty or 0.0),
            token_presence_penalty=(self.args.presence_penalty or 0.0),
        )
        return settings

    def generate_turn(self, turn_id: int, messages: list[dict[str, str]], emit: EventEmitter) -> None:
        emit(TurnStart(turn_id=turn_id))
        started = time.time()
        if self.logger is not None:
            self.logger.log(
                f"turn_start id={turn_id} messages={len(messages)} max_new_tokens={self.args.max_new_tokens} "
                f"max_seq_len={self.args.max_seq_len}",
                source="backend",
            )
        router = ThinkRouter(assume_think=self.args.assume_think)
        stage = "init"

        raw_parts: list[str] = []
        think_parts: list[str] = []
        answer_parts: list[str] = []

        try:
            stage = "template_select"
            if self.args.prompt_mode == "plain":
                template_text = DEFAULT_TEMPLATE
            else:
                template_text = resolve_chat_template(
                    self.args.chat_template,
                    model_dir=self.resolved_model_id,
                    config_path=getattr(self.args, "_config_path", None),
                )
                if not template_text:
                    template_text = DEFAULT_TEMPLATE

            stage = "context_encode"
            input_ids, trimmed_messages, prompt_text, context_report = apply_context_limit_exl2(
                self.runtime.tokenizer,
                template_text=template_text,
                messages=messages,
                max_seq_len=self.args.max_seq_len,
                min_free_tokens=self.args.min_free_tokens,
            )
            history_messages = list(trimmed_messages)

            prompt_tokens = _count_tokens(input_ids)
            emit(Meta(turn_id=turn_id, key="prompt_tokens", value=prompt_tokens))

            gen = self.runtime.generator
            settings = self._build_settings()
            stop_conditions = _build_stop_conditions(self.runtime.tokenizer, self.args)
            if stop_conditions:
                gen.set_stop_conditions(stop_conditions)
            knob_sent = {
                "max_new_tokens": self.args.max_new_tokens,
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "top_k": settings.top_k,
                "min_p": settings.min_p,
                "typical_p": settings.typical,
                "repetition_penalty": settings.token_repetition_penalty,
                "frequency_penalty": settings.token_frequency_penalty,
                "presence_penalty": settings.token_presence_penalty,
            }
            if self.args.stop_strings:
                knob_sent["stop_strings"] = list(self.args.stop_strings)
            knob_report = finalize_knob_report(
                sent=knob_sent,
                supported=SUPPORTED_KNOBS["exl2"],
                ignored=unsupported_user_set(self.args, "exl2"),
            )

            stage = "begin_stream"
            gen.begin_stream_ex(input_ids, settings)

            generated_tokens = 0
            context_restarts = 0
            repeat_last_token: int | None = None
            repeat_streak = 0
            while True:
                stage = "stream_loop"
                res = gen.stream_ex() or {}
                chunk = res.get("chunk", "") or ""
                eos = bool(res.get("eos"))
                token_ids = res.get("chunk_token_ids")
                ids = _iter_token_ids(token_ids)
                token_inc = len(ids) if ids else _count_tokens(token_ids)
                generated_tokens += token_inc
                if token_inc > 0:
                    emit(Meta(turn_id=turn_id, key="generated_tokens_inc", value=token_inc))

                repeat_limit = int(getattr(self.args, "exl2_repeat_streak_max", 64) or 0)
                if repeat_limit > 0 and ids:
                    for tid in ids:
                        if repeat_last_token == tid:
                            repeat_streak += 1
                        else:
                            repeat_last_token = tid
                            repeat_streak = 1
                        if repeat_streak >= repeat_limit:
                            emit(Meta(turn_id=turn_id, key="repeat_streak_stop", value=repeat_streak))
                            eos = True
                            break

                if chunk:
                    raw_parts.append(chunk)
                    for channel, text in router.feed(chunk):
                        if channel == "think":
                            think_parts.append(text)
                            emit(ThinkDelta(turn_id=turn_id, text=text))
                        else:
                            answer_parts.append(text)
                            emit(AnswerDelta(turn_id=turn_id, text=text))

                if eos:
                    break
                if self.args.max_new_tokens and generated_tokens >= self.args.max_new_tokens:
                    break
                full_fn = getattr(gen, "full", None)
                if callable(full_fn) and full_fn():
                    if context_restarts >= 2:
                        emit(Meta(turn_id=turn_id, key="context_restarts", value=context_restarts))
                        break
                    remaining = None
                    if self.args.max_new_tokens:
                        remaining = max(1, int(self.args.max_new_tokens) - int(generated_tokens))
                    reserve = self.args.min_free_tokens
                    if remaining is not None:
                        reserve = max(64, min(self.args.max_seq_len - 1, remaining))
                    resume_messages = list(history_messages)
                    partial = "".join(raw_parts)
                    if partial:
                        resume_messages.append({"role": "assistant", "content": partial})
                    stage = "context_rebuild"
                    input_ids, resume_trimmed, _ = apply_context_limit_exl2(
                        self.runtime.tokenizer,
                        template_text=template_text,
                        messages=resume_messages,
                        max_seq_len=self.args.max_seq_len,
                        min_free_tokens=reserve,
                    )
                    history_messages = _strip_trailing_assistant_if_matches(resume_trimmed, partial)
                    stage = "resume_stream"
                    gen.begin_stream_ex(input_ids, settings)
                    context_restarts += 1
                    emit(Meta(turn_id=turn_id, key="context_restarts", value=context_restarts))

        except Exception as exc:
            if self.logger is not None:
                self.logger.log(f"turn_error id={turn_id} stage={stage} error={exc}", source="backend")
            emit(Error(turn_id=turn_id, message=f"{stage}: {exc}"))
            return

        for channel, text in router.flush():
            if channel == "think":
                think_parts.append(text)
                emit(ThinkDelta(turn_id=turn_id, text=text))
            else:
                answer_parts.append(text)
                emit(AnswerDelta(turn_id=turn_id, text=text))

        ended = time.time()
        completion_tokens = int(generated_tokens)
        total_tokens = int(prompt_tokens) + completion_tokens
        emit(Meta(turn_id=turn_id, key="completion_tokens", value=completion_tokens))
        emit(Meta(turn_id=turn_id, key="total_tokens", value=total_tokens))
        elapsed = max(0.0, ended - started)
        throughput = None
        if completion_tokens > 0 and elapsed > 0:
            throughput = {"tokens_per_s": completion_tokens / elapsed}
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
                "min_p": self.args.min_p,
                "typical_p": self.args.typical_p,
                "repetition_penalty": self.args.repetition_penalty,
            },
            timing={"start": started, "end": ended, "elapsed": elapsed},
            token_counts={
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            throughput=throughput,
            knobs=knob_report,
            context=context_report,
            trimmed_messages=append_assistant_history(
                list(history_messages),
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


def _parse_size_to_bytes(value: str, unit: str) -> int:
    scale = {
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
    }
    return int(float(value) * scale[unit.lower()])


def _format_gib(value_bytes: int | None) -> str:
    if value_bytes is None:
        return "unknown"
    return f"{(value_bytes / (1024**3)):.2f} GiB"


def _build_gpu_oom_message(exc: Exception, *, max_seq_len: int, cache_type: str) -> str:
    raw = str(exc)
    lowered = raw.lower()
    if "out of memory" not in lowered and "cuda" not in lowered:
        return raw

    requested_bytes: int | None = None
    m = re.search(r"tried to allocate\s+([0-9.]+)\s*(kib|mib|gib|kb|mb|gb)", raw, re.IGNORECASE)
    if m:
        try:
            requested_bytes = _parse_size_to_bytes(m.group(1), m.group(2))
        except Exception:
            requested_bytes = None

    free_bytes: int | None = None
    total_bytes: int | None = None
    if torch.cuda.is_available():
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        except Exception:
            pass

    overflow_bytes: int | None = None
    if requested_bytes is not None and free_bytes is not None:
        overflow_bytes = max(0, requested_bytes - free_bytes)

    lines = [
        "EXL2 model load failed due to GPU memory pressure.",
        f"gpu_free: {_format_gib(free_bytes)} / gpu_total: {_format_gib(total_bytes)}",
        f"requested_allocation: {_format_gib(requested_bytes)}",
        f"estimated_overflow: {_format_gib(overflow_bytes)}",
        f"settings: max_seq_len={max_seq_len}, cache_type={cache_type}",
        "Try lowering max_seq_len, reducing cache precision, or using a smaller quant.",
        f"Original error: {raw}",
    ]
    return "\n".join(lines)


def _detect_runtime_attention_backend(model, config) -> str | None:
    # Runtime introspection only; no inference in this helper.
    for attr in ("attention_backend", "attn_backend", "backend"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    for attr in ("attention_backend", "attn_backend", "backend"):
        value = getattr(config, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _query_attention_capabilities() -> dict[str, object]:
    out: dict[str, object] = {
        "flash_attn_available": "unknown",
        "xformers_available": "unknown",
        "sdpa_available": "unknown",
    }
    try:
        from exllamav2 import attn as exl2_attn  # type: ignore

        out["flash_attn_available"] = bool(getattr(exl2_attn, "has_flash_attn", False))
        out["xformers_available"] = bool(getattr(exl2_attn, "has_xformers", False))
        out["sdpa_available"] = bool(getattr(exl2_attn, "has_lower_right_sdpa", False))
    except Exception:
        pass
    return out

def _effective_attention_target(config, caps: dict[str, object]) -> str:
    # Deterministic target based on user toggles + verified capability flags.
    flash_on = not bool(getattr(config, "no_flash_attn", False))
    xformers_on = not bool(getattr(config, "no_xformers", False))
    sdpa_on = not bool(getattr(config, "no_sdpa", False))
    if flash_on and caps.get("flash_attn_available") is True:
        return "flash_attn"
    if xformers_on and caps.get("xformers_available") is True:
        return "xformers"
    if sdpa_on and caps.get("sdpa_available") is True:
        return "sdpa"
    return "eager/torch"


def create_session(args: argparse.Namespace) -> EXL2Session:
    logger = FileLogger.from_value(
        getattr(args, "exl2_log_file", ""),
        "backend",
        config_path=getattr(args, "_config_path", None),
    )
    model_dir = resolve_exl2_model_dir(
        args.model_id,
        model_path=getattr(args, "model_path", None),
        exl2_repo_path=getattr(args, "exl2_repo_path", None),
        config_path=getattr(args, "_config_path", None),
    )
    if not os.path.isdir(model_dir):
        raise RuntimeError(
            f"EXL2 model_dir not found: {model_dir} "
            "(set model_id to full path, or set model_path/exl2_repo_path to a base directory containing the model folder)"
        )
    if logger is not None:
        logger.log(f"session_init model_dir={model_dir}", source="app")

    try:
        _try_import_exllamav2(getattr(args, "exl2_repo_path", None))
    except Exception as exc:
        msg = str(exc)
        hint = "Missing dependency: ExLlamaV2.\n"
        if "Ninja is required" in msg:
            hint += "Install build dep: .venv/bin/pip install ninja\n"
        hint += "See docs: docs/exl2_setup.md"
        raise RuntimeError(hint) from exc

    from exllamav2 import (
        ExLlamaV2,
        ExLlamaV2Cache,
        ExLlamaV2Cache_8bit,
        ExLlamaV2Cache_Q4,
        ExLlamaV2Cache_Q6,
        ExLlamaV2Cache_Q8,
        ExLlamaV2Config,
        ExLlamaV2Tokenizer,
    )
    from exllamav2.generator import ExLlamaV2StreamingGenerator

    config = ExLlamaV2Config()
    config.model_dir = model_dir
    config.prepare()
    if not torch.cuda.is_available():
        raise RuntimeError("EXL2 backend requires CUDA. CPU fallback is disabled for this backend.")

    if getattr(args, "seed", None) is not None:
        seed = int(args.seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if not getattr(args, "max_seq_len", None):
        args.max_seq_len = config.max_seq_len
    if args.max_seq_len:
        config.max_seq_len = args.max_seq_len
    if args.min_free_tokens < 1:
        args.min_free_tokens = 1
    if args.max_new_tokens:
        desired_free = min(int(args.max_new_tokens), max(1, int(config.max_seq_len) - 1))
        if args.min_free_tokens < desired_free:
            args.min_free_tokens = desired_free

    if args.rope_yarn:
        if getattr(config, "alt_rope_method", None) != "yarn":
            config.yarn_rope_original_max_position_embeddings = config.max_seq_len
        config.alt_rope_method = "yarn"
        config.yarn_rope_factor = args.rope_yarn

    if args.rope_scale is not None:
        config.scale_pos_emb = args.rope_scale
    if args.rope_alpha is not None:
        config.scale_alpha_value = args.rope_alpha

    # Attention toggles (tri-state).
    if args.flash_attn is False:
        config.no_flash_attn = True
    elif args.flash_attn is True:
        config.no_flash_attn = False
    if args.xformers is False:
        config.no_xformers = True
    elif args.xformers is True:
        config.no_xformers = False
    if args.sdpa is False:
        config.no_sdpa = True
    elif args.sdpa is True:
        config.no_sdpa = False
    if args.graphs is False:
        config.no_graphs = True
    elif args.graphs is True:
        config.no_graphs = False

    if args.low_mem:
        config.set_low_mem()

    # Compatibility overrides can fix common arch issues but can also be noisy; keep default behavior.
    config.arch_compat_overrides(warn_only=False)

    # Load model
    model = ExLlamaV2(config)

    cache_type = (args.cache_type or "fp16").lower()
    cache_cls = ExLlamaV2Cache
    if cache_type in {"8bit", "int8"}:
        cache_cls = ExLlamaV2Cache_8bit
    elif cache_type == "q4":
        cache_cls = ExLlamaV2Cache_Q4
    elif cache_type == "q6":
        cache_cls = ExLlamaV2Cache_Q6
    elif cache_type == "q8":
        cache_cls = ExLlamaV2Cache_Q8

    split = None
    if args.gpu_split and args.gpu_split != "auto":
        split = [float(x) for x in args.gpu_split.split(",")]
    try:
        model.load(split, progress=True)
        cache = cache_cls(model, max_seq_len=config.max_seq_len, lazy=False)
    except Exception as exc:
        raise RuntimeError(
            _build_gpu_oom_message(exc, max_seq_len=config.max_seq_len, cache_type=cache_type)
        ) from exc

    tokenizer = ExLlamaV2Tokenizer(config)
    generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
    runtime = _Exl2Runtime(
        model=model,
        cache=cache,
        tokenizer=tokenizer,
        generator=generator,
        backend_info=(
            lambda caps: {
            "attention_backend_runtime": (_detect_runtime_attention_backend(model, config) or "unknown"),
            "attention_backend_effective": _effective_attention_target(config, caps),
            "flash_attn_enabled": (not bool(getattr(config, "no_flash_attn", False))),
            "xformers_enabled": (not bool(getattr(config, "no_xformers", False))),
            "sdpa_enabled": (not bool(getattr(config, "no_sdpa", False))),
            "graphs_enabled": (not bool(getattr(config, "no_graphs", False))),
            "flash_attn_available": caps.get("flash_attn_available"),
            "xformers_available": caps.get("xformers_available"),
            "sdpa_available": caps.get("sdpa_available"),
        })(_query_attention_capabilities()),
    )
    if logger is not None:
        logger.log(
            "attention_runtime="
            f"{runtime.backend_info.get('attention_backend_runtime')} "
            f"attention_effective={runtime.backend_info.get('attention_backend_effective')}",
            source="app",
        )
    requested_template = (args.chat_template or "").strip()
    template_requested_value = requested_template
    if requested_template:
        try:
            candidate = resolve_path_maybe_relative(requested_template, config_path=getattr(args, "_config_path", None))
            if os.path.isfile(candidate):
                template_requested_value = candidate
        except Exception:
            pass
    template_info = {
        "chat_template_requested": template_requested_value,
        "chat_template_applied": bool(requested_template and args.prompt_mode != "plain"),
        "chat_template_reason": (
            "applied" if requested_template and args.prompt_mode != "plain" else ("ignored_prompt_mode" if requested_template else "empty_default")
        ),
    }
    return EXL2Session(runtime=runtime, args=args, resolved_model_id=model_dir, template_info=template_info, logger=logger)
