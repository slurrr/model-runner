from __future__ import annotations

import argparse
import json
import os
import queue
import re
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
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
from tui_app.events import AnswerDelta, Error, Finish, Meta, ThinkDelta, TurnRecord, TurnStart
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


def apply_context_limit(tokenizer, messages, max_context_tokens, prompt_mode, *, processor=None, multimodal: bool = False):
    if not max_context_tokens:
        if multimodal and processor is not None:
            inputs, _ = build_model_inputs_multimodal(processor=processor, messages=messages)
            return inputs, messages
        return build_model_inputs(tokenizer=tokenizer, messages=messages, prompt_mode=prompt_mode), messages

    trimmed = list(messages)
    while True:
        if multimodal and processor is not None:
            candidate_inputs, _ = build_model_inputs_multimodal(processor=processor, messages=trimmed)
        else:
            candidate_inputs = build_model_inputs(tokenizer=tokenizer, messages=trimmed, prompt_mode=prompt_mode)
        input_len = candidate_inputs["input_ids"].shape[-1]
        if input_len <= max_context_tokens:
            return candidate_inputs, trimmed

        drop_idx = None
        start_idx = 1 if trimmed and trimmed[0].get("role") == "system" else 0
        for idx in range(start_idx, len(trimmed)):
            if trimmed[idx].get("role") in {"user", "assistant", "tool"}:
                drop_idx = idx
                break
        if drop_idx is None:
            return candidate_inputs, trimmed
        trimmed.pop(drop_idx)


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
        logger: FileLogger | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.args = args
        self.input_device = input_device
        self.resolved_model_id = resolved_model_id
        self.supports_images = supports_images
        self.logger = logger

    def close(self) -> None:
        if self.logger is not None:
            self.logger.close()

    def get_recent_logs(self, n: int = 80) -> list[str]:
        if self.logger is None:
            return []
        return self.logger.get_recent_logs(n=n)

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
                    f"max_new_tokens={args.max_new_tokens}"
                )
            multimodal = bool(self.supports_images and any(msg.get("images") for msg in messages))
            if multimodal and self.processor is None:
                raise RuntimeError("Images were attached but this HF session has no processor loaded.")
            if multimodal and args.prompt_mode != "chat":
                raise RuntimeError("Images require --prompt-mode chat.")

            model_inputs, trimmed = apply_context_limit(
                tokenizer=tokenizer,
                messages=messages,
                max_context_tokens=args.max_context_tokens,
                prompt_mode=args.prompt_mode,
                processor=self.processor,
                multimodal=multimodal,
            )
            model_inputs = {
                key: value.to(self.input_device) if hasattr(value, "to") else value
                for key, value in model_inputs.items()
            }

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

                for piece in streamer:
                    while True:
                        try:
                            count = streamer.think_token_queue.get_nowait()
                        except queue.Empty:
                            break
                        if count is None:
                            break
                        emit(Meta(turn_id=turn_id, key="think_tokens_inc", value=int(count)))

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
            else:
                input_len = model_inputs["input_ids"].shape[-1]
                with torch.no_grad():
                    outputs = model.generate(**model_inputs, **generate_kwargs)
                new_tokens = outputs[0, input_len:].tolist()
                for token_id in new_tokens:
                    piece = tokenizer.decode(
                        [token_id],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    pre_mode = router.mode
                    is_marker = piece in router.start_markers or piece in router.end_markers
                    if pre_mode == "think" and not is_marker:
                        emit(Meta(turn_id=turn_id, key="think_tokens_inc", value=1))
                    raw_parts.append(piece)
                    for channel, text in router.feed(piece):
                        if channel == "think":
                            think_parts.append(text)
                            emit(ThinkDelta(turn_id=turn_id, text=text))
                        else:
                            answer_parts.append(text)
                            emit(AnswerDelta(turn_id=turn_id, text=text))

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
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "repetition_penalty": args.repetition_penalty,
                },
                timing={"start": started, "end": ended, "elapsed": max(0.0, ended - started)},
                trimmed_messages=trimmed,
            )
            emit(Finish(turn_id=turn_id, record=record))
            if self.logger is not None:
                self.logger.log(
                    f"turn_finish id={turn_id} elapsed_s={record.timing.get('elapsed', 0):.3f} "
                    f"think_chars={len(record.think)} answer_chars={len(record.answer)}"
                )
        except Exception as exc:
            if self.logger is not None:
                self.logger.log(f"turn_error id={turn_id} error={exc}")
            emit(Error(turn_id=turn_id, message=str(exc)))


def create_session(args: argparse.Namespace) -> HFSession:
    device = pick_default_device()
    dtype = resolve_dtype(args.dtype, device)

    resolved_model_id = resolve_model_id(args.model_id)
    logger = FileLogger.from_value(getattr(args, "hf_log_file", ""), "hf")
    print(f"Loading model: {resolved_model_id}")
    print(f"Using device={device}, dtype={dtype}")
    if logger is not None:
        logger.log(f"session_init model={resolved_model_id} device={device} dtype={dtype}")

    model_type = read_model_type(resolved_model_id)
    if model_type == "personaplex":
        raise RuntimeError(
            "This checkpoint is a speech-to-speech PersonaPlex model, not a text chat causal-LM checkpoint."
        )

    tokenizer = load_tokenizer(resolved_model_id)
    supports_images = is_vision_checkpoint(resolved_model_id)
    processor = None
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
        if tokenizer.chat_template is None:
            raise RuntimeError("Tokenizer has no chat template. Use --prompt-mode plain or set --chat-template.")

    model_kwargs = {}
    input_device = device

    if args.use_8bit or args.use_4bit:
        if device != "cuda":
            raise RuntimeError("4-bit/8-bit quantization requires CUDA.")
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = dtype
        if args.use_8bit:
            model_kwargs["load_in_8bit"] = True
            print("Quantization: 8-bit")
        else:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["bnb_4bit_compute_dtype"] = dtype
            print("Quantization: 4-bit")
        input_device = "cuda"
    else:
        model_kwargs["torch_dtype"] = dtype
        model_kwargs["device_map"] = "auto" if device == "cuda" else None
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
    if not (args.use_8bit or args.use_4bit):
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
        logger=logger,
    )
