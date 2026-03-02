import argparse
import json
import os
import re
import sys
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from config_utils import load_json_config


class ThinkFilter:
    def __init__(self, strict_prefix_strip: bool = False):
        self.start_end_map = {
            "<think>": "</think>",
            "<|begin_of_thought|>": "<|end_of_thought|>",
            "<｜begin_of_thought｜>": "<｜end_of_thought｜>",
            "<｜begin▁of▁thought｜>": "<｜end▁of▁thought｜>",
        }
        self.start_markers = list(self.start_end_map.keys())
        self.end_markers = list(set(self.start_end_map.values()) | {"</think>"})
        self.max_marker_len = max(len(x) for x in self.start_end_map) + 8
        self.in_think = False
        self.current_end_marker = ""
        self.buffer = ""
        # Some models emit hidden reasoning without an opening marker and only end with </think>.
        # Hold early output briefly to detect and strip that prefix if present.
        self.implicit_prefix_mode = True
        self.implicit_prefix_buffer = ""
        self.implicit_prefix_probe_limit = None if strict_prefix_strip else 8192

    @staticmethod
    def _find_first(text: str, markers: list[str]) -> tuple[int, str]:
        first_idx = -1
        first_marker = ""
        for marker in markers:
            idx = text.find(marker)
            if idx != -1 and (first_idx == -1 or idx < first_idx):
                first_idx = idx
                first_marker = marker
        return first_idx, first_marker

    def feed(self, text: str) -> str:
        if self.implicit_prefix_mode:
            self.implicit_prefix_buffer += text
            end_idx, end_marker = self._find_first(self.implicit_prefix_buffer, self.end_markers)
            if end_idx != -1:
                remainder = self.implicit_prefix_buffer[end_idx + len(end_marker) :]
                self.implicit_prefix_mode = False
                self.implicit_prefix_buffer = ""
                if not remainder:
                    return ""
                return self.feed(remainder)

            if self.implicit_prefix_probe_limit is None:
                return ""

            if len(self.implicit_prefix_buffer) <= self.implicit_prefix_probe_limit:
                return ""

            release = self.implicit_prefix_buffer
            self.implicit_prefix_mode = False
            self.implicit_prefix_buffer = ""
            return self.feed(release)

        self.buffer += text
        out = []

        while True:
            if not self.in_think:
                start_idx, start_marker = self._find_first(self.buffer, self.start_markers)
                if start_idx == -1:
                    keep = self.max_marker_len
                    if len(self.buffer) > keep:
                        out.append(self.buffer[:-keep])
                        self.buffer = self.buffer[-keep:]
                    break

                out.append(self.buffer[:start_idx])
                self.buffer = self.buffer[start_idx + len(start_marker) :]
                self.in_think = True
                self.current_end_marker = self.start_end_map[start_marker]
            else:
                end_idx = self.buffer.find(self.current_end_marker)
                if end_idx == -1:
                    keep = max(self.max_marker_len, len(self.current_end_marker) + 8)
                    if len(self.buffer) > keep:
                        self.buffer = self.buffer[-keep:]
                    break

                self.buffer = self.buffer[end_idx + len(self.current_end_marker) :]
                self.in_think = False
                self.current_end_marker = ""

        return "".join(out)

    def flush(self) -> str:
        if self.implicit_prefix_mode:
            remaining = self.implicit_prefix_buffer
            self.implicit_prefix_mode = False
            self.implicit_prefix_buffer = ""
            return remaining
        if self.in_think:
            self.buffer = ""
            self.current_end_marker = ""
            return ""
        remaining = self.buffer
        self.buffer = ""
        return remaining


def strip_think_text(text: str, strict_prefix_strip: bool = False) -> str:
    filter_state = ThinkFilter(strict_prefix_strip=strict_prefix_strip)
    out = filter_state.feed(text)
    out += filter_state.flush()
    return out


def pick_default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_dtype(dtype_name):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def resolve_dtype(dtype_name, device):
    if dtype_name == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    return parse_dtype(dtype_name)


def read_model_type(model_id):
    if os.path.isdir(model_id):
        config_path = os.path.join(model_id, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as fh:
                    return json.load(fh).get("model_type")
            except Exception:
                return None
    return None


def resolve_model_id(model_id):
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


def load_tokenizer(model_id):
    try:
        return AutoTokenizer.from_pretrained(model_id)
    except Exception:
        return AutoTokenizer.from_pretrained(model_id, use_fast=False)


def resolve_chat_template(template_spec, model_id):
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

    path = os.path.expanduser(spec)
    if not os.path.isabs(path):
        path = os.path.abspath(path)
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


def apply_context_limit(tokenizer, messages, max_context_tokens):
    if not max_context_tokens:
        templated = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        if isinstance(templated, torch.Tensor):
            return {"input_ids": templated}, messages
        return dict(templated), messages

    # Keep system prompt and most recent turns until within budget.
    trimmed = list(messages)
    while True:
        templated = tokenizer.apply_chat_template(
            trimmed,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        if isinstance(templated, torch.Tensor):
            candidate_inputs = {"input_ids": templated}
        else:
            candidate_inputs = dict(templated)

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


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="")
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="Simple local HF chat runner")
    parser.add_argument("model_id", nargs="?", help="Hugging Face model ID or local path")
    parser.add_argument("--config", default="", help="Config path or name (e.g. models/Nanbeige4.1-3B/hf/config).")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--stream", action="store_true", help="Stream assistant output token-by-token.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--typical-p", type=float, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--stop-strings", nargs="+", default=None)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Model precision. 'auto' picks float16 on CUDA, float32 on CPU.",
    )
    parser.add_argument(
        "--hide-think",
        action="store_true",
        help="Strip common <think> / thought-markers from assistant messages.",
    )
    parser.add_argument(
        "--strict-think-strip",
        action="store_true",
        help="If set, strip everything up to the first closing thought marker (more aggressive).",
    )
    parser.add_argument(
        "--chat-template",
        default="",
        help="Template selector: default | search | <path-to-.jinja/.txt/.json>",
    )
    parser.add_argument(
        "--system",
        default="",
        help="Optional system prompt added once at the beginning.",
    )
    parser.add_argument(
        "--system-file",
        default="",
        help="Optional text file for system prompt (used if --system is empty).",
    )
    parser.add_argument(
        "--user-prefix",
        default="",
        help="Optional prefix prepended to each user turn.",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="If set, trim oldest conversation turns to keep prompt length under this token count.",
    )
    quant_group = parser.add_mutually_exclusive_group()
    quant_group.add_argument(
        "-8bit",
        "--8bit",
        dest="use_8bit",
        action="store_true",
        help="Load model with bitsandbytes 8-bit quantization",
    )
    quant_group.add_argument(
        "-4bit",
        "--4bit",
        dest="use_4bit",
        action="store_true",
        help="Load model with bitsandbytes 4-bit quantization",
    )

    config_data = {}
    if pre_args.config:
        try:
            config_data, resolved = load_json_config(pre_args.config, backend="hf")
            print(f"Loaded config: {resolved}")
        except Exception as exc:
            print(f"Failed to load config: {exc}")
            sys.exit(1)

    supported_keys = {
        "model_id",
        "max_new_tokens",
        "stream",
        "temperature",
        "top_p",
        "top_k",
        "typical_p",
        "min_p",
        "repetition_penalty",
        "max_time",
        "num_beams",
        "no_repeat_ngram_size",
        "stop_strings",
        "dtype",
        "hide_think",
        "strict_think_strip",
        "system",
        "chat_template",
        "system_file",
        "user_prefix",
        "max_context_tokens",
        "use_8bit",
        "use_4bit",
    }
    config_defaults = {k: v for k, v in config_data.items() if k in supported_keys}
    if config_defaults:
        parser.set_defaults(**config_defaults)

    args = parser.parse_args()
    if not args.model_id:
        args.model_id = config_defaults.get("model_id")
    if not args.model_id:
        parser.error("model_id is required (or provide model_id in --config).")

    args.model_id = resolve_model_id(args.model_id)
    if not args.system and args.system_file:
        try:
            with open(os.path.expanduser(args.system_file), "r", encoding="utf-8") as fh:
                args.system = fh.read().strip()
        except Exception as exc:
            print(f"Failed to read system file '{args.system_file}': {exc}")
            sys.exit(1)

    device = pick_default_device()
    dtype = resolve_dtype(args.dtype, device)
    print(f"Loading model: {args.model_id}")
    print(f"Using device={device}, dtype={dtype}")

    model_type = read_model_type(args.model_id)
    if model_type == "personaplex":
        print(
            "This checkpoint is a speech-to-speech PersonaPlex model, not a text "
            "chat causal-LM checkpoint. Use NVIDIA PersonaPlex/Moshi inference instead."
        )
        sys.exit(1)

    try:
        tokenizer = load_tokenizer(args.model_id)
        template_override = resolve_chat_template(args.chat_template, args.model_id)
        if template_override is not None:
            tokenizer.chat_template = template_override
            print(f"Chat template override: {args.chat_template}")

        model_kwargs = {}
        input_device = device

        if args.use_8bit or args.use_4bit:
            if device != "cuda":
                print("4-bit/8-bit quantization requires CUDA.")
                sys.exit(1)
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

        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
        if not (args.use_8bit or args.use_4bit):
            model.to(device)
        model.eval()
    except Exception as exc:
        print(f"Failed to load model/tokenizer: {exc}")
        print("If tokenizer conversion fails, install: sentencepiece, tiktoken, protobuf")
        sys.exit(1)

    if tokenizer.chat_template is None:
        print("Tokenizer has no chat template. Use runner.py or a chat model/tokenizer.")
        sys.exit(1)

    messages = []
    if args.system.strip():
        messages.append({"role": "system", "content": args.system.strip()})

    print("Chat ready. Commands: /exit, /quit, /clear")
    while True:
        try:
            user_text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_text.lower() in {"/exit", "/quit", "exit", "quit"}:
            break
        if user_text.lower() == "/clear":
            messages = []
            if args.system.strip():
                messages.append({"role": "system", "content": args.system.strip()})
            print("Conversation cleared.")
            continue
        if not user_text:
            continue

        rendered_user = f"{args.user_prefix}{user_text}" if args.user_prefix else user_text
        messages.append({"role": "user", "content": rendered_user})
        model_inputs, trimmed_messages = apply_context_limit(
            tokenizer=tokenizer,
            messages=messages,
            max_context_tokens=args.max_context_tokens,
        )
        if trimmed_messages is not messages:
            messages = trimmed_messages

        model_inputs = {
            key: value.to(input_device) if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }

        input_len = model_inputs["input_ids"].shape[-1]
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
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            stream_kwargs = dict(generate_kwargs)
            stream_kwargs["streamer"] = streamer

            raw_parts = []
            shown_parts = []
            think_filter = ThinkFilter(strict_prefix_strip=args.strict_think_strip) if args.hide_think else None

            def _generate():
                with torch.no_grad():
                    model.generate(**model_inputs, **stream_kwargs)

            thread = threading.Thread(target=_generate, daemon=True)
            thread.start()

            for piece in streamer:
                raw_parts.append(piece)
                if think_filter is not None:
                    filtered = think_filter.feed(piece)
                    if filtered:
                        shown_parts.append(filtered)
                        print(filtered, end="", flush=True)
                else:
                    shown_parts.append(piece)
                    print(piece, end="", flush=True)

            if think_filter is not None:
                tail = think_filter.flush()
                if tail:
                    shown_parts.append(tail)
                    print(tail, end="", flush=True)

            thread.join()
            print()
            raw_text = "".join(raw_parts).strip()
            shown_text = "".join(shown_parts).strip()
        else:
            with torch.no_grad():
                outputs = model.generate(
                    **model_inputs,
                    **generate_kwargs,
                )

            new_tokens = outputs[0, input_len:]
            raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            shown_text = raw_text
            if args.hide_think:
                shown_text = strip_think_text(raw_text, strict_prefix_strip=args.strict_think_strip).strip()
            print(shown_text)

        messages.append({"role": "assistant", "content": shown_text})


if __name__ == "__main__":
    main()
