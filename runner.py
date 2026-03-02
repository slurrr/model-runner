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


def choose_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


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


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="")
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="Simple local HF model runner")
    parser.add_argument("model_id", nargs="?", help="Hugging Face model ID, e.g. gpt2")
    parser.add_argument("--config", default="", help="Config path or name (e.g. models/Nanbeige4.1-3B/hf/config).")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--stream", action="store_true", help="Stream generated output token-by-token.")
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
        "--prompt-prefix",
        default="",
        help="Optional prefix prepended to each user prompt before tokenization.",
    )
    parser.add_argument(
        "--hide-think",
        action="store_true",
        help="Strip common <think> / thought-markers from decoded output.",
    )
    parser.add_argument(
        "--strict-think-strip",
        action="store_true",
        help="If set, strip everything up to the first closing thought marker (more aggressive).",
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
        "prompt_prefix",
        "hide_think",
        "strict_think_strip",
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

    device, dtype = choose_device_and_dtype()
    print(f"Loading model: {args.model_id}")
    print(f"Using device={device}, dtype={dtype}")

    model_type = read_model_type(args.model_id)
    if model_type == "personaplex":
        print(
            "This checkpoint is a speech-to-speech PersonaPlex model, not a text "
            "causal-LM checkpoint. Use NVIDIA PersonaPlex/Moshi inference instead."
        )
        sys.exit(1)

    try:
        tokenizer = load_tokenizer(args.model_id)
        model_kwargs = {}
        input_device = device

        if args.use_8bit or args.use_4bit:
            if device != "cuda":
                print("4-bit/8-bit quantization requires CUDA.")
                sys.exit(1)
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.float16
            if args.use_8bit:
                model_kwargs["load_in_8bit"] = True
                print("Quantization: 8-bit")
            else:
                model_kwargs["load_in_4bit"] = True
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

    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue

        rendered_prompt = f"{args.prompt_prefix}{prompt}" if args.prompt_prefix else prompt
        inputs = tokenizer(rendered_prompt, return_tensors="pt").to(input_device)
        input_len = inputs["input_ids"].shape[-1]

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

            shown_parts = []
            think_filter = ThinkFilter(strict_prefix_strip=args.strict_think_strip) if args.hide_think else None

            def _generate():
                with torch.no_grad():
                    model.generate(**inputs, **stream_kwargs)

            thread = threading.Thread(target=_generate, daemon=True)
            thread.start()

            for piece in streamer:
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
        else:
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    **generate_kwargs,
                )

            new_tokens = output[0, input_len:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
            if args.hide_think:
                decoded = strip_think_text(decoded, strict_prefix_strip=args.strict_think_strip)
            print(decoded)


if __name__ == "__main__":
    main()
