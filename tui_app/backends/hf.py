from __future__ import annotations

import argparse
import json
import os
import queue
import re
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer

from tui_app.backends.base import EventEmitter
from tui_app.events import AnswerDelta, Error, Finish, Meta, ThinkDelta, TurnRecord, TurnStart
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
        content = (msg.get("content", "") or "").strip()
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

    templated = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    if isinstance(templated, torch.Tensor):
        return {"input_ids": templated}
    return dict(templated)


def apply_context_limit(tokenizer, messages, max_context_tokens, prompt_mode):
    if not max_context_tokens:
        return build_model_inputs(tokenizer=tokenizer, messages=messages, prompt_mode=prompt_mode), messages

    trimmed = list(messages)
    while True:
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
    def __init__(self, tokenizer, skip_prompt: bool = True, timeout: float | None = None):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.timeout = timeout
        self.text_queue = queue.Queue()
        self.stop_signal = None
        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True
        self.mode = "answer"
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

    def __init__(self, model, tokenizer, args: argparse.Namespace, input_device: str, resolved_model_id: str):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.input_device = input_device
        self.resolved_model_id = resolved_model_id

    def generate_turn(self, turn_id: int, messages: list[dict[str, str]], emit: EventEmitter) -> None:
        args = self.args
        tokenizer = self.tokenizer
        model = self.model
        emit(TurnStart(turn_id=turn_id))

        started = time.time()
        router = ThinkRouter()
        raw_parts = []
        think_parts = []
        answer_parts = []

        try:
            model_inputs, trimmed = apply_context_limit(
                tokenizer=tokenizer,
                messages=messages,
                max_context_tokens=args.max_context_tokens,
                prompt_mode=args.prompt_mode,
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
                streamer = TokenCountingTextIteratorStreamer(tokenizer, skip_prompt=True)
                generate_kwargs["streamer"] = streamer

                def _generate():
                    with torch.no_grad():
                        model.generate(**model_inputs, **generate_kwargs)

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
        except Exception as exc:
            emit(Error(turn_id=turn_id, message=str(exc)))


def create_session(args: argparse.Namespace) -> HFSession:
    device = pick_default_device()
    dtype = resolve_dtype(args.dtype, device)

    resolved_model_id = resolve_model_id(args.model_id)
    print(f"Loading model: {resolved_model_id}")
    print(f"Using device={device}, dtype={dtype}")

    model_type = read_model_type(resolved_model_id)
    if model_type == "personaplex":
        raise RuntimeError(
            "This checkpoint is a speech-to-speech PersonaPlex model, not a text chat causal-LM checkpoint."
        )

    tokenizer = load_tokenizer(resolved_model_id)
    if args.prompt_mode == "chat":
        template_override = resolve_chat_template(args.chat_template, resolved_model_id, config_path=args._config_path)
        if template_override is not None:
            tokenizer.chat_template = template_override
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

    model = AutoModelForCausalLM.from_pretrained(resolved_model_id, **model_kwargs)
    if not (args.use_8bit or args.use_4bit):
        model.to(device)
    model.eval()
    return HFSession(model=model, tokenizer=tokenizer, args=args, input_device=input_device, resolved_model_id=resolved_model_id)
