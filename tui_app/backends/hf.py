from __future__ import annotations

import argparse
import importlib.util
import json
import os
import queue
import re
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None  # type: ignore
from transformers.generation.streamers import BaseStreamer

try:
    from transformers import AutoModelForMultimodalLM  # type: ignore
except Exception:
    AutoModelForMultimodalLM = None  # type: ignore

try:
    from transformers import AutoModelForImageTextToText  # type: ignore
except Exception:
    AutoModelForImageTextToText = None  # type: ignore

from tui_app.backends.base import EventEmitter
from tui_app.context_policy import build_context_limit_error, reserve_generation_tokens, trim_messages_to_budget
from tui_app.events import AnswerDelta, Error, Finish, Meta, ThinkDelta, TurnRecord, TurnStart
from tui_app.history_control import append_assistant_history
from tui_app.knobs import SUPPORTED_KNOBS, finalize_knob_report, unsupported_user_set
from tui_app.log_file import FileLogger
from tui_app.think_router import ThinkRouter


def pick_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_dtype(dtype_name: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def resolve_dtype(dtype_name: str, device: str):
    if dtype_name == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    return parse_dtype(dtype_name)


def _parse_cuda_index(device_str: str | None) -> int | None:
    if not device_str:
        return None
    raw = str(device_str)
    if raw == "cuda":
        return 0
    if raw.startswith("cuda:"):
        try:
            return int(raw.split(":", 1)[1])
        except Exception:
            return None
    return None


def resolve_model_id(model_id: str) -> str:
    raw = model_id.strip()
    expanded = os.path.expanduser(raw)

    win_drive = re.match(r"^([A-Za-z]):[\\/](.*)$", expanded)
    if win_drive:
        drive = win_drive.group(1).lower()
        rest = win_drive.group(2).replace("\\", "/")
        expanded = f"/mnt/{drive}/{rest}"

    if os.path.exists(expanded):
        return os.path.abspath(expanded)

    if raw and not ("/" in raw or raw.startswith(".") or raw.startswith("~")):
        local_models = os.path.expanduser(os.path.join("~", "ml", "models", raw))
        if os.path.exists(local_models):
            return os.path.abspath(local_models)

    return model_id


def read_model_type(model_id: str):
    if os.path.isdir(model_id):
        config_path = os.path.join(model_id, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as fh:
                    return json.load(fh).get("model_type")
            except Exception:
                return None
    return None


def is_vision_checkpoint(model_id: str) -> bool:
    if not os.path.isdir(model_id):
        return False
    config_path = os.path.join(model_id, "config.json")
    if not os.path.isfile(config_path):
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return False
    # Common patterns: vision token IDs and/or a vision_config sub-config.
    if "vision_start_token_id" in data or "vision_end_token_id" in data:
        return True
    vision_cfg = data.get("vision_config")
    if isinstance(vision_cfg, dict) and vision_cfg:
        return True
    return False


def load_tokenizer(model_id: str):
    try:
        return AutoTokenizer.from_pretrained(model_id)
    except Exception:
        return AutoTokenizer.from_pretrained(model_id, use_fast=False)


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


def resolve_chat_template(template_spec: str, model_id: str, config_path: str | None = None):
    if not template_spec:
        return None

    spec = template_spec.strip()
    lowered = spec.lower()
    if lowered in {"default", "tokenizer_config"}:
        return None

    model_dir = model_id if os.path.isdir(model_id) else None
    if lowered in {"search", "tokenizer_config_search"}:
        if not model_dir:
            raise ValueError("chat_template 'search' requires a local model directory.")
        candidate = os.path.join(model_dir, "tokenizer_config_search.json")
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"Template source not found: {candidate}")
        with open(candidate, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        template = data.get("chat_template")
        if not template:
            raise ValueError(f"'chat_template' missing in {candidate}")
        return template

    path = resolve_path_maybe_relative(spec, config_path=config_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Template file not found: {path}")

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        template = data.get("chat_template") or data.get("template")
        if not template:
            raise ValueError(f"No 'chat_template' or 'template' field in {path}")
        return template

    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def render_plain_prompt(messages):
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content_obj = msg.get("content", "")
        content = content_obj if isinstance(content_obj, str) else str(content_obj)
        content = content.strip()
        if not content:
            continue
        if role == "system":
            parts.append(f"System: {content}\n")
        elif role == "user":
            parts.append(f"User: {content}\n")
        elif role == "assistant":
            parts.append(f"Assistant: {content}\n")
        elif role == "tool":
            parts.append(f"Tool: {content}\n")
        else:
            parts.append(f"{role.title() if role else 'Message'}: {content}\n")
    parts.append("Assistant:")
    return "\n".join(parts)


def _parse_hf_max_memory(raw: str):
    value = (raw or "").strip()
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except Exception:
        return {0: value}
    if isinstance(parsed, dict):
        return parsed
    return {0: value}


def build_model_inputs(tokenizer, messages, prompt_mode):
    if prompt_mode == "plain":
        prompt = render_plain_prompt(messages)
        return tokenizer(prompt, return_tensors="pt")

    # Drop non-standard keys (e.g. "images") so apply_chat_template sees a plain chat history.
    sanitized = []
    for msg in messages:
        content_obj = msg.get("content", "")
        content = content_obj if isinstance(content_obj, str) else str(content_obj)
        sanitized.append({"role": msg.get("role", ""), "content": content})

    templated = tokenizer.apply_chat_template(
        sanitized,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    if isinstance(templated, torch.Tensor):
        return {"input_ids": templated}
    return dict(templated)


def _open_images(image_paths: list[str]):
    from PIL import Image

    images = []
    for path in image_paths:
        img = Image.open(path)
        images.append(img.convert("RGB"))
    return images


def _build_multimodal_messages(messages: list[dict[str, object]]):
    """
    Convert our internal message shape (role/content + optional images=[...paths...])
    into the multimodal message format expected by Qwen-style processor chat templates.
    Returns (messages_mm, image_paths_in_order).
    """
    messages_mm: list[dict[str, object]] = []
    image_paths: list[str] = []
    for msg in messages:
        role = str(msg.get("role", ""))
        content_obj = msg.get("content", "")
        text = content_obj if isinstance(content_obj, str) else str(content_obj)
        images = msg.get("images")
        if isinstance(images, list) and role == "user":
            segments: list[dict[str, object]] = []
            for path in images:
                if not isinstance(path, str):
                    continue
                image_paths.append(path)
                # The Qwen chat template only checks type; the actual image tensor is supplied via processor(..., images=...).
                segments.append({"type": "image", "image": path})
            if text:
                segments.append({"type": "text", "text": text})
            messages_mm.append({"role": role, "content": segments})
        else:
            messages_mm.append({"role": role, "content": text})
    return messages_mm, image_paths


def build_model_inputs_multimodal(processor, messages: list[dict[str, object]]):
    messages_mm, image_paths = _build_multimodal_messages(messages)
    prompt = processor.apply_chat_template(messages_mm, tokenize=False, add_generation_prompt=True)
    if image_paths:
        images = _open_images(image_paths)
        return processor(text=prompt, images=images, return_tensors="pt"), messages_mm
    return processor(text=prompt, return_tensors="pt"), messages_mm


def _infer_context_window(model, tokenizer) -> int | None:
    candidates = [
        getattr(getattr(model, "config", None), "max_position_embeddings", None),
        getattr(getattr(model, "config", None), "n_positions", None),
        getattr(getattr(model, "config", None), "max_seq_len", None),
        getattr(getattr(model, "config", None), "seq_length", None),
        getattr(getattr(getattr(model, "config", None), "text_config", None), "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    ]
    for value in candidates:
        if isinstance(value, int) and 0 < value < 1_000_000:
            return int(value)
    return None


def _normalize_eos_token_ids(eos_token_id) -> set[int]:
    if isinstance(eos_token_id, int):
        return {int(eos_token_id)}
    if isinstance(eos_token_id, (list, tuple, set)):
        return {int(value) for value in eos_token_id if isinstance(value, int)}
    return set()


def _infer_hf_finish_reason(
    *,
    generated_token_ids: list[int],
    completion_tokens: int,
    max_new_tokens: int | None,
    eos_token_id,
) -> tuple[str | None, str | None]:
    if isinstance(max_new_tokens, int) and max_new_tokens > 0 and completion_tokens >= max_new_tokens:
        return "length", "local_max_new_tokens_cap"
    eos_token_ids = _normalize_eos_token_ids(eos_token_id)
    if generated_token_ids and eos_token_ids and int(generated_token_ids[-1]) in eos_token_ids:
        return "stop", "local_eos_token"
    if completion_tokens > 0:
        return "stop", "local_hf_stopping_criteria"
    return None, None


def apply_context_limit(
    tokenizer,
    messages,
    max_context_tokens,
    prompt_mode,
    *,
    max_new_tokens: int | None,
    model=None,
    processor=None,
    multimodal: bool = False,
):
    context_window = int(max_context_tokens) if max_context_tokens else _infer_context_window(model, tokenizer)
    if not context_window:
        if multimodal and processor is not None:
            inputs, _ = build_model_inputs_multimodal(processor=processor, messages=messages)
            return inputs, messages, None
        return build_model_inputs(tokenizer=tokenizer, messages=messages, prompt_mode=prompt_mode), messages, None

    reserved = reserve_generation_tokens(context_window, max_new_tokens)

    def _measure(trimmed):
        if multimodal and processor is not None:
            candidate_inputs, _ = build_model_inputs_multimodal(processor=processor, messages=trimmed)
        else:
            candidate_inputs = build_model_inputs(tokenizer=tokenizer, messages=trimmed, prompt_mode=prompt_mode)
        input_len = candidate_inputs["input_ids"].shape[-1]
        return int(input_len), candidate_inputs

    trimmed, _prompt_tokens, model_inputs, report = trim_messages_to_budget(
        list(messages),
        measure_fn=_measure,
        context_window=context_window,
        reserved_generation_tokens=reserved,
        strategy="exact_preflight",
    )
    if not report.fit:
        raise RuntimeError(build_context_limit_error(report))
    return model_inputs, trimmed, report.to_dict()


class TokenCountingTextIteratorStreamer(BaseStreamer):
    def __init__(
        self,
        tokenizer,
        skip_prompt: bool = True,
        timeout: float | None = None,
        assume_think: bool = False,
    ):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.timeout = timeout
        self.text_queue = queue.Queue()
        self.stop_signal = None
        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True
        self.generated_token_count = 0
        self.generated_token_ids = []
        self.mode = "think" if assume_think else "answer"
        self.start_marker_ids = self._single_token_marker_ids(
            ["<think>", "<|begin_of_thought|>", "<｜begin_of_thought｜>", "<｜begin▁of▁thought｜>"]
        )
        self.end_marker_ids = self._single_token_marker_ids(
            ["</think>", "<|end_of_thought|>", "<｜end_of_thought｜>", "<｜end▁of▁thought｜>"]
        )
        self.think_token_queue = queue.Queue()

    def _single_token_marker_ids(self, markers):
        ids = set()
        for marker in markers:
            token_ids = self.tokenizer.encode(marker, add_special_tokens=False)
            if len(token_ids) == 1:
                ids.add(token_ids[0])
        return ids

    @staticmethod
    def _is_chinese_char(cp):
        return (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)
            or (cp >= 0x20000 and cp <= 0x2A6DF)
            or (cp >= 0x2A700 and cp <= 0x2B73F)
            or (cp >= 0x2B740 and cp <= 0x2B81F)
            or (cp >= 0x2B820 and cp <= 0x2CEAF)
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)
        )

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TokenCountingTextIteratorStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        token_ids = value.tolist()
        self.generated_token_count += len(token_ids)
        self.generated_token_ids.extend(int(token_id) for token_id in token_ids)
        think_inc = 0
        for token_id in token_ids:
            if token_id in self.start_marker_ids:
                self.mode = "think"
                continue
            if token_id in self.end_marker_ids:
                self.mode = "answer"
                continue
            if self.mode == "think":
                think_inc += 1
        if think_inc > 0:
            self.think_token_queue.put(think_inc)

        self.token_cache.extend(token_ids)
        text = self.tokenizer.decode(
            self.token_cache,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        self.text_queue.put(printable_text, timeout=self.timeout)

    def end(self):
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(
                self.token_cache,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""

        self.next_tokens_are_prompt = True
        self.text_queue.put(printable_text, timeout=self.timeout)
        self.text_queue.put(self.stop_signal, timeout=self.timeout)
        self.think_token_queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration
        return value


class HFSession:
    backend_name = "hf"

    def __init__(
        self,
        model,
        tokenizer,
        args: argparse.Namespace,
        input_device: str,
        resolved_model_id: str,
        *,
        processor=None,
        supports_images: bool = False,
        template_info: dict[str, object] | None = None,
        logger: FileLogger | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.args = args
        self.input_device = input_device
        self.resolved_model_id = resolved_model_id
        self.supports_images = supports_images
        self.template_info = dict(template_info or {})
        self.logger = logger

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
        weights_quantization = "8bit" if bool(getattr(self.args, "use_8bit", False)) else "4bit" if bool(getattr(self.args, "use_4bit", False)) else "none"
        model = self.model
        context_length_effective = _infer_context_window(model, self.tokenizer)
        runtime_attn = getattr(getattr(model, "config", None), "_attn_implementation", None)
        runtime_dtype = None
        runtime_device = None
        try:
            first_param = next(model.parameters())
            runtime_dtype = str(first_param.dtype).replace("torch.", "")
            runtime_device = str(first_param.device)
        except Exception:
            runtime_dtype = str(getattr(self.args, "dtype", "auto"))
            runtime_device = self.input_device

        raw_device_map = getattr(model, "hf_device_map", None)
        requested_device_map = getattr(self.args, "hf_device_map", "") or ("auto" if self.input_device == "cuda" else "none")
        if isinstance(raw_device_map, dict):
            device_map_repr = dict(raw_device_map)
            map_values = [str(v) for v in raw_device_map.values()]
            modules_on_cpu = sum(1 for v in map_values if v == "cpu")
            modules_on_disk = sum(1 for v in map_values if v == "disk")
            gpu_targets = sorted({v for v in map_values if v.startswith("cuda") or v.isdigit()})
            fully_on_single_gpu = modules_on_cpu == 0 and modules_on_disk == 0 and len(gpu_targets) == 1
        else:
            device_map_repr = requested_device_map
            normalized_target = str(runtime_device)
            fully_on_single_gpu = normalized_target.startswith("cuda")
            modules_on_cpu = 0 if fully_on_single_gpu else None
            modules_on_disk = 0

        memory_footprint_bytes = None
        try:
            memory_footprint_bytes = int(model.get_memory_footprint())
        except Exception:
            memory_footprint_bytes = None

        if memory_footprint_bytes is None:
            memory_footprint = "unavailable"
        else:
            memory_footprint = f"{memory_footprint_bytes / (1024 ** 3):.2f} GiB"

        cuda_allocated = None
        cuda_reserved = None
        cuda_max_allocated = None
        cuda_max_reserved = None
        cuda_index = _parse_cuda_index(runtime_device)
        if cuda_index is not None and torch.cuda.is_available():
            try:
                cuda_allocated = int(torch.cuda.memory_allocated(cuda_index))
                cuda_reserved = int(torch.cuda.memory_reserved(cuda_index))
                cuda_max_allocated = int(torch.cuda.max_memory_allocated(cuda_index))
                cuda_max_reserved = int(torch.cuda.max_memory_reserved(cuda_index))
            except Exception:
                pass

        def _fmt_bytes(value: int | None) -> str:
            if value is None:
                return "unavailable"
            return f"{value / (1024 ** 3):.2f} GiB"

        max_memory_raw = getattr(self.args, "hf_max_memory", "") or "(unset)"
        max_memory_effective = bool(str(requested_device_map).lower() in {"auto", "balanced", "balanced_low_0", "sequential"})

        qwen_fast_path_available = None
        if read_model_type(self.resolved_model_id) == "qwen3_5":
            has_fla = importlib.util.find_spec("fla") is not None
            has_causal_conv1d = importlib.util.find_spec("causal_conv1d") is not None
            qwen_fast_path_available = bool(has_fla and has_causal_conv1d)

        return {
            "template_control_level": "local_template",
            "supports_images": self.supports_images,
            "weights_quantization": weights_quantization,
            "kv_cache_quantization": "none",
            "torch_dtype": str(getattr(self.args, "dtype", "auto")),
            "torch_dtype_effective": runtime_dtype,
            "hf_attn_implementation": getattr(self.args, "hf_attn_implementation", None) or "default",
            "attention_backend_effective": runtime_attn or "unknown",
            "hf_device_map": device_map_repr,
            "hf_max_memory": max_memory_raw,
            "hf_max_memory_effective": max_memory_effective,
            "hf_low_cpu_mem_usage": getattr(self.args, "hf_low_cpu_mem_usage", None),
            "runtime_device": runtime_device,
            "context_length_effective": context_length_effective,
            "modules_on_cpu": modules_on_cpu,
            "modules_on_disk": modules_on_disk,
            "fully_on_single_gpu": fully_on_single_gpu,
            "memory_footprint": memory_footprint,
            "memory_footprint_bytes": memory_footprint_bytes,
            "cuda_memory_allocated": _fmt_bytes(cuda_allocated),
            "cuda_memory_reserved": _fmt_bytes(cuda_reserved),
            "cuda_max_memory_allocated": _fmt_bytes(cuda_max_allocated),
            "cuda_max_memory_reserved": _fmt_bytes(cuda_max_reserved),
            "qwen_fast_path_available": qwen_fast_path_available,
            "context_allocation": "grows_with_sequence",
            "text_only_mode": bool(getattr(self.args, "hf_text_only", False)),
            **self.template_info,
        }

    def generate_turn(self, turn_id: int, messages: list[dict[str, object]], emit: EventEmitter) -> None:
        args = self.args
        tokenizer = self.tokenizer
        model = self.model
        emit(TurnStart(turn_id=turn_id))

        started = time.time()
        router = ThinkRouter(assume_think=args.assume_think)
        raw_parts = []
        think_parts = []
        answer_parts = []

        try:
            if self.logger is not None:
                self.logger.log(
                    f"turn_start id={turn_id} messages={len(messages)} stream={bool(args.stream)} "
                    f"max_new_tokens={args.max_new_tokens}",
                    source="backend",
                )
            multimodal = bool(self.supports_images and any(msg.get("images") for msg in messages))
            if multimodal and self.processor is None:
                raise RuntimeError("Images were attached but this HF session has no processor loaded.")
            if multimodal and args.prompt_mode != "chat":
                raise RuntimeError("Images require --prompt-mode chat.")

            model_inputs, trimmed, context_report = apply_context_limit(
                tokenizer=tokenizer,
                messages=messages,
                max_context_tokens=args.max_context_tokens,
                prompt_mode=args.prompt_mode,
                max_new_tokens=args.max_new_tokens,
                model=model,
                processor=self.processor,
                multimodal=multimodal,
            )
            model_inputs = {
                key: value.to(self.input_device) if hasattr(value, "to") else value
                for key, value in model_inputs.items()
            }
            prompt_tokens = int(model_inputs["input_ids"].shape[-1])
            emit(Meta(turn_id=turn_id, key="prompt_tokens", value=prompt_tokens))

            pad_token_id = tokenizer.eos_token_id
            if pad_token_id is None and tokenizer.pad_token_id is not None:
                pad_token_id = tokenizer.pad_token_id

            generate_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "do_sample": args.temperature > 0,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "pad_token_id": pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "num_beams": args.num_beams,
                "no_repeat_ngram_size": args.no_repeat_ngram_size,
            }
            if args.top_k is not None:
                generate_kwargs["top_k"] = args.top_k
            if args.typical_p is not None:
                generate_kwargs["typical_p"] = args.typical_p
            if args.min_p is not None:
                generate_kwargs["min_p"] = args.min_p
            if args.max_time is not None:
                generate_kwargs["max_time"] = args.max_time
            if args.stop_strings:
                generate_kwargs["stop_strings"] = args.stop_strings
                generate_kwargs["tokenizer"] = tokenizer

            knob_sent = {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "num_beams": args.num_beams,
                "no_repeat_ngram_size": args.no_repeat_ngram_size,
            }
            if "top_k" in generate_kwargs:
                knob_sent["top_k"] = args.top_k
            if "typical_p" in generate_kwargs:
                knob_sent["typical_p"] = args.typical_p
            if "min_p" in generate_kwargs:
                knob_sent["min_p"] = args.min_p
            if "max_time" in generate_kwargs:
                knob_sent["max_time"] = args.max_time
            if "stop_strings" in generate_kwargs:
                knob_sent["stop_strings"] = args.stop_strings
            knob_report = finalize_knob_report(
                sent=knob_sent,
                supported=SUPPORTED_KNOBS["hf"],
                ignored=unsupported_user_set(args, "hf"),
            )

            if args.stream:
                streamer = TokenCountingTextIteratorStreamer(
                    tokenizer,
                    skip_prompt=True,
                    assume_think=args.assume_think,
                )
                generate_kwargs["streamer"] = streamer
                gen_error: list[Exception] = []

                def _generate():
                    try:
                        with torch.no_grad():
                            model.generate(**model_inputs, **generate_kwargs)
                    except Exception as exc:
                        gen_error.append(exc)
                        # Ensure consumer iterator unblocks even on generation failure.
                        try:
                            streamer.end()
                        except Exception:
                            pass

                gen_thread = threading.Thread(target=_generate, daemon=True)
                gen_thread.start()

                emitted_generated_tokens = 0
                for piece in streamer:
                    current_generated_tokens = int(getattr(streamer, "generated_token_count", 0))
                    token_inc = current_generated_tokens - emitted_generated_tokens
                    if token_inc > 0:
                        emit(Meta(turn_id=turn_id, key="generated_tokens_inc", value=token_inc))
                        emitted_generated_tokens = current_generated_tokens

                    raw_parts.append(piece)
                    for channel, text in router.feed(piece):
                        if channel == "think":
                            think_parts.append(text)
                            emit(ThinkDelta(turn_id=turn_id, text=text))
                        else:
                            answer_parts.append(text)
                            emit(AnswerDelta(turn_id=turn_id, text=text))

                gen_thread.join()
                if gen_error:
                    raise RuntimeError(str(gen_error[0]))
                completion_tokens = int(getattr(streamer, "generated_token_count", 0))
                generated_token_ids = [int(token_id) for token_id in getattr(streamer, "generated_token_ids", [])]
            else:
                input_len = model_inputs["input_ids"].shape[-1]
                with torch.no_grad():
                    outputs = model.generate(**model_inputs, **generate_kwargs)
                new_tokens = outputs[0, input_len:].tolist()
                completion_tokens = len(new_tokens)
                generated_token_ids = [int(token_id) for token_id in new_tokens]
                for token_id in new_tokens:
                    emit(Meta(turn_id=turn_id, key="generated_tokens_inc", value=1))
                    piece = tokenizer.decode(
                        [token_id],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    raw_parts.append(piece)
                    for channel, text in router.feed(piece):
                        if channel == "think":
                            think_parts.append(text)
                            emit(ThinkDelta(turn_id=turn_id, text=text))
                        else:
                            answer_parts.append(text)
                            emit(AnswerDelta(turn_id=turn_id, text=text))

            total_tokens = prompt_tokens + int(completion_tokens)
            emit(Meta(turn_id=turn_id, key="completion_tokens", value=int(completion_tokens)))
            emit(Meta(turn_id=turn_id, key="total_tokens", value=total_tokens))

            for channel, text in router.flush():
                if channel == "think":
                    think_parts.append(text)
                    emit(ThinkDelta(turn_id=turn_id, text=text))
                else:
                    answer_parts.append(text)
                    emit(AnswerDelta(turn_id=turn_id, text=text))

            ended = time.time()
            elapsed = max(0.0, ended - started)
            throughput = None
            if completion_tokens > 0 and elapsed > 0:
                throughput = {"tokens_per_s": completion_tokens / elapsed}
            finish_reason, finish_reason_source = _infer_hf_finish_reason(
                generated_token_ids=generated_token_ids,
                completion_tokens=int(completion_tokens),
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
            )
            answer_text = "".join(answer_parts)
            gen = {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
            }
            if finish_reason is not None:
                gen["finish_reason"] = finish_reason
            if finish_reason_source is not None:
                gen["finish_reason_source"] = finish_reason_source
            record = TurnRecord(
                raw="".join(raw_parts),
                think="".join(think_parts),
                answer=answer_text,
                ended_in_think=(router.mode == "think"),
                backend=self.backend_name,
                model_id=self.resolved_model_id,
                gen=gen,
                timing={"start": started, "end": ended, "elapsed": elapsed},
                token_counts={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": int(completion_tokens),
                    "total_tokens": total_tokens,
                },
                throughput=throughput,
                knobs=knob_report,
                context=context_report,
                trimmed_messages=append_assistant_history(
                    list(trimmed),
                    think="".join(think_parts),
                    answer=answer_text,
                    strip_think=bool(args.history_strip_think),
                ),
            )
            emit(Finish(turn_id=turn_id, record=record))
            if self.logger is not None:
                self.logger.log(
                    f"turn_finish id={turn_id} elapsed_s={record.timing.get('elapsed', 0):.3f} "
                    f"think_chars={len(record.think)} answer_chars={len(record.answer)}",
                    source="backend",
                )
        except Exception as exc:
            if self.logger is not None:
                self.logger.log(f"turn_error id={turn_id} error={exc}", source="backend")
            emit(Error(turn_id=turn_id, message=str(exc)))


def create_session(args: argparse.Namespace) -> HFSession:
    device = pick_default_device()
    dtype = resolve_dtype(args.dtype, device)

    resolved_model_id = resolve_model_id(args.model_id)
    logger = FileLogger.from_value(getattr(args, "hf_log_file", ""), "backend", config_path=getattr(args, "_config_path", None))
    print(f"Loading model: {resolved_model_id}")
    print(f"Using device={device}, dtype={dtype}")
    if logger is not None:
        logger.log(f"session_init model={resolved_model_id} device={device} dtype={dtype}", source="app")

    model_type = read_model_type(resolved_model_id)
    if model_type == "personaplex":
        raise RuntimeError(
            "This checkpoint is a speech-to-speech PersonaPlex model, not a text chat causal-LM checkpoint."
        )

    tokenizer = load_tokenizer(resolved_model_id)
    supports_images = is_vision_checkpoint(resolved_model_id) and not bool(getattr(args, "hf_text_only", False))
    processor = None
    requested_template = (args.chat_template or "").strip()
    template_requested_value = requested_template
    if requested_template:
        try:
            candidate = resolve_path_maybe_relative(requested_template, config_path=args._config_path)
            if os.path.isfile(candidate):
                template_requested_value = candidate
        except Exception:
            pass
    template_info = {
        "chat_template_requested": template_requested_value,
        "chat_template_applied": False,
        "chat_template_reason": "empty_default" if not requested_template else "ignored_prompt_mode",
    }
    if is_vision_checkpoint(resolved_model_id) and bool(getattr(args, "hf_text_only", False)):
        template_info["text_only_mode"] = True
    else:
        template_info["text_only_mode"] = False
    if supports_images:
        try:
            processor = AutoProcessor.from_pretrained(resolved_model_id)
        except Exception as exc:
            raise RuntimeError(f"Failed to load processor for vision checkpoint: {exc}") from exc
    if args.prompt_mode == "chat":
        template_override = resolve_chat_template(args.chat_template, resolved_model_id, config_path=args._config_path)
        if template_override is not None:
            tokenizer.chat_template = template_override
            if processor is not None and hasattr(processor, "tokenizer") and getattr(processor, "tokenizer", None) is not None:
                try:
                    processor.tokenizer.chat_template = template_override
                except Exception:
                    pass
            print(f"Chat template override: {args.chat_template}")
            template_info = {
                "chat_template_requested": template_requested_value,
                "chat_template_applied": True,
                "chat_template_reason": "applied",
            }
        elif not requested_template:
            template_info = {
                "chat_template_requested": template_requested_value,
                "chat_template_applied": False,
                "chat_template_reason": "empty_default",
            }
        if tokenizer.chat_template is None:
            raise RuntimeError("Tokenizer has no chat template. Use --prompt-mode plain or set --chat-template.")
    elif requested_template:
        template_info = {
            "chat_template_requested": template_requested_value,
            "chat_template_applied": False,
            "chat_template_reason": "ignored_prompt_mode",
        }

    model_kwargs = {}
    input_device = device
    explicit_device_map = (getattr(args, "hf_device_map", "") or "").strip()
    parsed_max_memory = _parse_hf_max_memory(getattr(args, "hf_max_memory", ""))
    explicit_low_cpu_mem_usage = getattr(args, "hf_low_cpu_mem_usage", None)

    if args.use_8bit or args.use_4bit:
        if device != "cuda":
            raise RuntimeError("4-bit/8-bit quantization requires CUDA.")
        if BitsAndBytesConfig is None:
            raise RuntimeError("BitsAndBytesConfig is unavailable in this transformers build.")
        model_kwargs["device_map"] = explicit_device_map or "auto"
        model_kwargs["torch_dtype"] = dtype
        if parsed_max_memory is not None:
            model_kwargs["max_memory"] = parsed_max_memory
        if explicit_low_cpu_mem_usage is not None:
            model_kwargs["low_cpu_mem_usage"] = bool(explicit_low_cpu_mem_usage)
        if args.use_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            print("Quantization: 8-bit")
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
            )
            print("Quantization: 4-bit")
        input_device = "cuda"
    else:
        model_kwargs["torch_dtype"] = dtype
        model_kwargs["device_map"] = explicit_device_map or ("auto" if device == "cuda" else None)
        if parsed_max_memory is not None:
            model_kwargs["max_memory"] = parsed_max_memory
        if explicit_low_cpu_mem_usage is not None:
            model_kwargs["low_cpu_mem_usage"] = bool(explicit_low_cpu_mem_usage)
    if args.hf_attn_implementation:
        model_kwargs["attn_implementation"] = args.hf_attn_implementation
        print(f"HF attention implementation override: {args.hf_attn_implementation}")

    if supports_images:
        load_errors = []
        model = None
        if AutoModelForMultimodalLM is not None:
            try:
                model = AutoModelForMultimodalLM.from_pretrained(resolved_model_id, **model_kwargs)
            except Exception as exc:
                load_errors.append(f"AutoModelForMultimodalLM: {exc}")
        if model is None and AutoModelForImageTextToText is not None:
            try:
                model = AutoModelForImageTextToText.from_pretrained(resolved_model_id, **model_kwargs)
            except Exception as exc:
                load_errors.append(f"AutoModelForImageTextToText: {exc}")
        if model is None:
            detail = "; ".join(load_errors) if load_errors else "no compatible multimodal auto model class available"
            raise RuntimeError(f"Failed to load vision model with installed transformers version: {detail}")
    else:
        model = AutoModelForCausalLM.from_pretrained(resolved_model_id, **model_kwargs)
    if not (args.use_8bit or args.use_4bit) and not model_kwargs.get("device_map"):
        model.to(device)
    model.eval()
    return HFSession(
        model=model,
        tokenizer=tokenizer,
        args=args,
        input_device=input_device,
        resolved_model_id=resolved_model_id,
        processor=processor,
        supports_images=supports_images,
        template_info=template_info,
        logger=logger,
    )
