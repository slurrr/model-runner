import argparse
import json
import os
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass

import torch
from config_utils import load_json_config
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.events import MouseScrollDown, MouseScrollUp
from textual.reactive import reactive
from textual.widgets import Input, Static
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer


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


def load_tokenizer(model_id):
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


def resolve_chat_template(template_spec, model_id, config_path: str | None = None):
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


class StreamingThinkParser:
    def __init__(self):
        self.start_markers = [
            "<think>",
            "<|begin_of_thought|>",
            "<｜begin_of_thought｜>",
            "<｜begin▁of▁thought｜>",
        ]
        self.end_markers = [
            "</think>",
            "<|end_of_thought|>",
            "<｜end_of_thought｜>",
            "<｜end▁of▁thought｜>",
        ]
        self.mode = "answer"
        self.buffer = ""
        all_markers = self.start_markers + self.end_markers
        self.max_marker_len = max(len(m) for m in all_markers)

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

    def feed(self, piece: str):
        self.buffer += piece
        events = []

        while True:
            if self.mode == "answer":
                idx, marker = self._find_first(self.buffer, self.start_markers)
                if idx == -1:
                    keep = self.max_marker_len
                    if len(self.buffer) > keep:
                        emit = self.buffer[:-keep]
                        if emit:
                            events.append(("answer", emit))
                        self.buffer = self.buffer[-keep:]
                    break

                before = self.buffer[:idx]
                if before:
                    events.append(("answer", before))
                self.buffer = self.buffer[idx + len(marker) :]
                self.mode = "think"
            else:
                idx, marker = self._find_first(self.buffer, self.end_markers)
                if idx == -1:
                    keep = self.max_marker_len
                    if len(self.buffer) > keep:
                        emit = self.buffer[:-keep]
                        if emit:
                            events.append(("think", emit))
                        self.buffer = self.buffer[-keep:]
                    break

                before = self.buffer[:idx]
                if before:
                    events.append(("think", before))
                self.buffer = self.buffer[idx + len(marker) :]
                self.mode = "answer"

        return events

    def flush(self):
        events = []
        if self.buffer:
            events.append((self.mode, self.buffer))
        self.buffer = ""
        return events


class TokenCountingTextIteratorStreamer(BaseStreamer):
    """
    TextIterator-like streamer that keeps readable chunking and tracks exact token counts
    inside <think> blocks via token IDs when markers are single-token.
    """

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

    def _emit_text(self, text: str, stream_end: bool = False):
        self.text_queue.put(text, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def _emit_think_tokens(self, count: int):
        if count > 0:
            self.think_token_queue.put(count)

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
        self._emit_think_tokens(think_inc)

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

        self._emit_text(printable_text, stream_end=False)

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
        self._emit_text(printable_text, stream_end=True)
        self.think_token_queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration
        return value


class UserMessage(Static):
    def __init__(self, text: str):
        super().__init__(classes="user-message")
        self.text = text

    def on_mount(self):
        self.update(Text(f"You: {self.text}", style="white"))


class ThinkingPanel(Container):
    expanded = reactive(False)
    running = reactive(True)
    phase = reactive(0)

    def __init__(self, start_expanded: bool, animate: bool):
        super().__init__(classes="thinking-panel")
        self.expanded = start_expanded
        self.animate = animate
        self.thinking_text = ""
        self.thinking_tokens = 0
        self.started_at = time.time()
        self.total_seconds = 0.0
        self.tokens_per_sec = 0.0
        self.ended_in_think = False

    def compose(self) -> ComposeResult:
        yield Static(id="thinking-header")
        yield Static(id="thinking-body")

    def on_mount(self):
        self._render_header()
        self._render_body()
        if self.animate:
            self.set_interval(0.12, self._tick)

    def _tick(self):
        if self.running and not self.expanded:
            self.phase += 1
            self._render_header()

    def _header_text(self) -> Text:
        arrow = "▼" if self.expanded else ">"
        base = "thinking..."
        elapsed = time.time() - self.started_at if self.running else self.total_seconds
        suffix = f" ({self.thinking_tokens} tokens"
        if self.running:
            suffix += f", {elapsed:.1f}s"
        else:
            suffix += f", {self.total_seconds:.1f}s, {self.tokens_per_sec:.1f} tok/sec"
        if self.ended_in_think:
            suffix += ", truncated"
        suffix += ")"

        header = f"{arrow} {base}{suffix}"
        text = Text(header, style="grey")

        if self.running and self.animate and not self.expanded:
            start = 2 + (self.phase % max(1, len(base)))
            end = min(start + 3, 2 + len(base))
            text.stylize("white", start, end)

        return text

    def _render_header(self):
        header = self.query_one("#thinking-header", Static)
        header.update(self._header_text())

    def _render_body(self):
        body = self.query_one("#thinking-body", Static)
        if self.expanded:
            body.styles.display = "block"
            body.styles.height = "auto"
            body.update(Text(self.thinking_text, style="grey"))
        else:
            body.styles.display = "none"
            body.update("")
        self.refresh(layout=True)

    def add_think_tokens(self, token_inc: int):
        if token_inc <= 0:
            return
        self.thinking_tokens += token_inc
        self._render_header()

    def append_thinking(self, text: str):
        if not text:
            return
        self.thinking_text += text
        self._render_header()
        if self.expanded:
            self._render_body()
        self.refresh(layout=True)

    def finish(self, ended_in_think: bool):
        self.running = False
        self.total_seconds = max(0.0, time.time() - self.started_at)
        self.tokens_per_sec = self.thinking_tokens / self.total_seconds if self.total_seconds > 0 else 0.0
        self.ended_in_think = ended_in_think
        self._render_header()

    def toggle(self):
        self.expanded = not self.expanded
        self._render_header()
        self._render_body()
        self.refresh(layout=True)

    def on_click(self, event):
        if event.widget.id == "thinking-header":
            self.toggle()


class AssistantMessage(Container):
    def __init__(self, start_expanded: bool, animate_thinking: bool):
        super().__init__(classes="assistant-message")
        self.start_expanded = start_expanded
        self.animate_thinking = animate_thinking
        self.answer_text = ""

    def compose(self) -> ComposeResult:
        self.thinking_panel = ThinkingPanel(start_expanded=self.start_expanded, animate=self.animate_thinking)
        self.answer_widget = Static(classes="assistant-answer")
        self.hint_widget = Static(classes="assistant-hint")
        yield self.thinking_panel
        yield self.answer_widget
        yield self.hint_widget

    def add_think_tokens(self, token_inc: int):
        self.thinking_panel.add_think_tokens(token_inc=token_inc)

    def append_think(self, text: str):
        self.thinking_panel.append_thinking(text)

    def append_answer(self, text: str):
        if not text:
            return
        self.answer_text += text
        self.answer_widget.styles.height = "auto"
        self.answer_widget.update(Text(self.answer_text, style="white"))
        self.refresh(layout=True)

    def finish(self, ended_in_think: bool):
        self.thinking_panel.finish(ended_in_think=ended_in_think)
        if not self.answer_text.strip():
            msg = "(No final answer produced; generation ended during thinking. Increase max_new_tokens or adjust prompt.)"
            self.hint_widget.update(Text(msg, style="yellow"))

    def toggle_thinking(self):
        self.thinking_panel.toggle()


class TranscriptPane(VerticalScroll):
    def _at_bottom(self, eps: float = 0.1) -> bool:
        return float(self.scroll_y) >= float(self.max_scroll_y) - eps

    def _break_follow(self):
        if hasattr(self.app, "_break_follow"):
            self.app._break_follow()

    def _resume_follow(self, immediate: bool = False):
        if hasattr(self.app, "_resume_follow"):
            self.app._resume_follow(immediate=immediate)

    def action_scroll_up(self):
        self._break_follow()
        return super().action_scroll_up()

    def action_page_up(self):
        self._break_follow()
        return super().action_page_up()

    def action_scroll_home(self):
        self._break_follow()
        return super().action_scroll_home()

    def action_scroll_down(self):
        result = super().action_scroll_down()
        if self._at_bottom():
            self._resume_follow()
        return result

    def action_page_down(self):
        result = super().action_page_down()
        if self._at_bottom():
            self._resume_follow()
        return result

    def action_scroll_end(self):
        result = super().action_scroll_end()
        self._resume_follow(immediate=True)
        return result

    def on_mouse_scroll_up(self, event: MouseScrollUp):
        self.action_scroll_up()
        event.stop()

    def on_mouse_scroll_down(self, event: MouseScrollDown):
        self.action_scroll_down()
        event.stop()


@dataclass
class TuiRuntime:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    args: argparse.Namespace
    input_device: str


class TuiChatApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    #transcript {
        height: 1fr;
        padding: 1 1;
        layout: vertical;
    }

    .user-message {
        margin: 0 0 1 0;
        height: auto;
    }

    .assistant-message {
        margin: 0 0 1 0;
        layout: vertical;
        height: auto;
    }

    .thinking-panel {
        border: none;
        margin: 0 0 1 0;
        layout: vertical;
        height: auto;
    }

    #thinking-header {
        color: grey;
        text-style: bold;
    }

    #thinking-body {
        color: grey;
        margin: 0 0 0 2;
        height: auto;
        width: 100%;
        text-wrap: wrap;
    }

    .assistant-answer {
        color: white;
        height: auto;
        width: 100%;
        text-wrap: wrap;
    }

    .assistant-hint {
        margin: 1 0 0 0;
    }

    #input-band {
        dock: bottom;
        height: 3;
        background: #3a3a3a;
        padding: 0 1;
    }

    #chat-input {
        width: 100%;
        background: #3a3a3a;
        color: white;
        border: none;
    }
    """

    BINDINGS = [
        Binding("t", "toggle_latest_thinking", "Toggle thinking"),
        Binding("pageup", "scroll_page_up", "Scroll up", priority=True),
        Binding("pagedown", "scroll_page_down", "Scroll down", priority=True),
        Binding("home", "scroll_home", "Scroll top", priority=True),
        Binding("end", "scroll_end_manual", "Scroll bottom", priority=True),
    ]

    def __init__(self, runtime: TuiRuntime):
        super().__init__()
        self.runtime = runtime
        self.messages = []
        if runtime.args.system:
            self.messages.append({"role": "system", "content": runtime.args.system})

        self.event_queue: queue.Queue = queue.Queue()
        self.pending_assistant: AssistantMessage | None = None
        self.pending_turn_id = 0
        self.is_generating = False
        self.turn_records = []
        self.follow_output = True
        self._scroll_end_scheduled = False
        self._max_events_per_tick = 240

    def compose(self) -> ComposeResult:
        self.transcript = TranscriptPane(id="transcript")
        yield self.transcript
        with Container(id="input-band"):
            yield Input(placeholder="Type message and press Enter", id="chat-input")

    def on_mount(self):
        self.transcript.can_focus = True
        self.set_interval(0.05, self._drain_events)
        self.call_after_refresh(self._scroll_to_end_now)

    def action_toggle_latest_thinking(self):
        if self.pending_assistant is not None:
            self.pending_assistant.toggle_thinking()

    def _break_follow(self):
        self.follow_output = False
        # Cancel pending auto-follow callbacks requested before user broke follow.
        self._scroll_end_scheduled = False

    def _resume_follow(self, immediate: bool = False):
        self.follow_output = True
        if immediate:
            self._scroll_end_scheduled = False
            self.transcript.refresh(layout=True)
            self.transcript.scroll_end(animate=False)
            return
        self._request_scroll_end()

    def action_scroll_page_up(self):
        self.transcript.action_page_up()

    def action_scroll_page_down(self):
        self.transcript.action_page_down()

    def action_scroll_home(self):
        self.transcript.action_scroll_home()

    def action_scroll_end_manual(self):
        self.transcript.action_scroll_end()

    def _scroll_to_end_now(self):
        self._scroll_end_scheduled = False
        if not self.follow_output:
            return
        self.transcript.refresh(layout=True)
        self.transcript.scroll_end(animate=False)

    def _request_scroll_end(self):
        if not self.follow_output:
            return
        if self._scroll_end_scheduled:
            return
        self._scroll_end_scheduled = True
        self.call_after_refresh(self._scroll_to_end_now)

    def _should_autofollow(self) -> bool:
        return self.follow_output

    async def on_input_submitted(self, event: Input.Submitted):
        text = event.value.strip()
        if not text:
            event.input.value = ""
            return
        if self.is_generating:
            self.notify("Generation in progress. Wait for current turn to finish.")
            return
        if text.lower() in {"/exit", "exit", "/quit", "quit"}:
            self.exit()
            return
        if text.lower() == "/clear":
            self.transcript.remove_children()
            self.messages = []
            if self.runtime.args.system:
                self.messages.append({"role": "system", "content": self.runtime.args.system})
            event.input.value = ""
            return

        user_text = f"{self.runtime.args.user_prefix}{text}" if self.runtime.args.user_prefix else text
        self.messages.append({"role": "user", "content": user_text})
        await self.transcript.mount(UserMessage(text))

        assistant = AssistantMessage(
            start_expanded=self.runtime.args.show_thinking,
            animate_thinking=not self.runtime.args.no_animate_thinking,
        )
        await self.transcript.mount(assistant)
        self.pending_assistant = assistant
        self.pending_turn_id += 1
        turn_id = self.pending_turn_id
        self.is_generating = True

        self.follow_output = True
        self.transcript.refresh(layout=True)
        self._request_scroll_end()
        event.input.value = ""

        thread = threading.Thread(target=self._run_generation, args=(turn_id,), daemon=True)
        thread.start()

    def _run_generation(self, turn_id: int):
        args = self.runtime.args
        tokenizer = self.runtime.tokenizer
        model = self.runtime.model
        input_device = self.runtime.input_device

        try:
            model_inputs, trimmed = apply_context_limit(
                tokenizer=tokenizer,
                messages=self.messages,
                max_context_tokens=args.max_context_tokens,
            )
            self.messages = trimmed

            model_inputs = {
                key: value.to(input_device) if hasattr(value, "to") else value
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

            parser = StreamingThinkParser()
            raw_parts = []
            answer_parts = []
            think_parts = []

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
                        self.event_queue.put((turn_id, "think_tokens", int(count)))

                    raw_parts.append(piece)
                    events = parser.feed(piece)
                    for channel, text in events:
                        if channel == "think":
                            think_parts.append(text)
                        else:
                            answer_parts.append(text)
                        self.event_queue.put((turn_id, channel, text))

                gen_thread.join()
                while True:
                    try:
                        count = streamer.think_token_queue.get_nowait()
                    except queue.Empty:
                        break
                    if count is None:
                        break
                    self.event_queue.put((turn_id, "think_tokens", int(count)))
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
                    pre_mode = parser.mode
                    is_marker = piece in parser.start_markers or piece in parser.end_markers
                    if pre_mode == "think" and not is_marker:
                        self.event_queue.put((turn_id, "think_tokens", 1))

                    raw_parts.append(piece)
                    events = parser.feed(piece)
                    for channel, text in events:
                        if channel == "think":
                            think_parts.append(text)
                        else:
                            answer_parts.append(text)
                        self.event_queue.put((turn_id, channel, text))

            for channel, text in parser.flush():
                if channel == "think":
                    think_parts.append(text)
                else:
                    answer_parts.append(text)
                self.event_queue.put((turn_id, channel, text))

            ended_in_think = parser.mode == "think"
            self.event_queue.put(
                (
                    turn_id,
                    "finish",
                    {
                        "raw": "".join(raw_parts),
                        "answer": "".join(answer_parts).strip(),
                        "think": "".join(think_parts),
                        "ended_in_think": ended_in_think,
                    },
                )
            )
        except Exception as exc:
            self.event_queue.put((turn_id, "error", str(exc)))

    def _drain_events(self):
        processed = 0
        pending_think_token_inc = 0
        while True:
            if processed >= self._max_events_per_tick:
                break
            try:
                turn_id, kind, payload = self.event_queue.get_nowait()
            except queue.Empty:
                break
            processed += 1

            if turn_id != self.pending_turn_id or self.pending_assistant is None:
                continue

            if kind == "think_tokens":
                pending_think_token_inc += int(payload)
                continue

            if pending_think_token_inc:
                self.pending_assistant.add_think_tokens(pending_think_token_inc)
                pending_think_token_inc = 0

            if kind == "think":
                self.pending_assistant.append_think(payload)
                if self._should_autofollow():
                    self._request_scroll_end()
            elif kind == "answer":
                self.pending_assistant.append_answer(payload)
                if self._should_autofollow():
                    self._request_scroll_end()
            elif kind == "error":
                self.pending_assistant.append_answer(f"\n[Generation error] {payload}")
                self.pending_assistant.finish(ended_in_think=False)
                self.is_generating = False
            elif kind == "finish":
                record = payload
                self.pending_assistant.finish(ended_in_think=record["ended_in_think"])
                assistant_for_history = record["answer"] if record["answer"] else record["think"]
                self.messages.append({"role": "assistant", "content": assistant_for_history})
                self.turn_records.append(record)
                if self.runtime.args.save_transcript:
                    self._append_transcript_record(record)
                self.is_generating = False
                if self._should_autofollow():
                    self._request_scroll_end()

        if pending_think_token_inc and self.pending_assistant is not None:
            self.pending_assistant.add_think_tokens(pending_think_token_inc)
        if processed > 0:
            self.transcript.refresh(layout=True)

    def _append_transcript_record(self, record):
        path = resolve_path_maybe_relative(self.runtime.args.save_transcript, config_path=self.runtime.args._config_path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="")
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="Textual TUI chat for local HF models")
    parser.add_argument("model_id", nargs="?", help="Hugging Face model ID or local path")
    parser.add_argument("--config", default="", help="Config path or name (e.g. Nanbeige4.1-3B).")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--stream", dest="stream", action="store_true", help="Stream assistant output.")
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="Disable streaming.")
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
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--chat-template", default="", help="Template selector: default | search | <path>")
    parser.add_argument("--system", default="", help="Optional system prompt")
    parser.add_argument("--system-file", default="", help="Optional system prompt file")
    parser.add_argument("--user-prefix", default="", help="Optional prefix prepended to each user turn")
    parser.add_argument("--max-context-tokens", type=int, default=None)
    parser.add_argument("--show-thinking", action="store_true", help="Start with thinking panels expanded")
    parser.add_argument("--no-animate-thinking", action="store_true", help="Disable thinking header animation")
    parser.add_argument("--save-transcript", default="", help="Optional JSONL path to save raw/parsed per-turn output")
    parser.add_argument("-8bit", "--8bit", dest="use_8bit", action="store_true")
    parser.add_argument("-4bit", "--4bit", dest="use_4bit", action="store_true")

    config_data = {}
    config_path = None
    if pre_args.config:
        try:
            config_data, config_path = load_json_config(pre_args.config, backend="hf")
            print(f"Loaded config: {config_path}")
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
        "chat_template",
        "system",
        "system_file",
        "user_prefix",
        "max_context_tokens",
        "show_thinking",
        "no_animate_thinking",
        "save_transcript",
        "use_8bit",
        "use_4bit",
    }
    config_defaults = {k: v for k, v in config_data.items() if k in supported_keys}
    if config_defaults:
        parser.set_defaults(**config_defaults)

    parser.set_defaults(stream=True)
    args = parser.parse_args()
    if not args.model_id:
        args.model_id = config_defaults.get("model_id")
    if not args.model_id:
        parser.error("model_id is required (or provide model_id in --config).")

    args._config_path = config_path
    args.model_id = resolve_model_id(args.model_id)
    if not args.system and args.system_file:
        try:
            path = resolve_path_maybe_relative(args.system_file, config_path=config_path)
            with open(path, "r", encoding="utf-8") as fh:
                args.system = fh.read().strip()
        except Exception as exc:
            print(f"Failed to read system file '{args.system_file}': {exc}")
            sys.exit(1)

    return args


def load_runtime(args: argparse.Namespace) -> TuiRuntime:
    device = pick_default_device()
    dtype = resolve_dtype(args.dtype, device)
    print(f"Loading model: {args.model_id}")
    print(f"Using device={device}, dtype={dtype}")

    model_type = read_model_type(args.model_id)
    if model_type == "personaplex":
        print(
            "This checkpoint is a speech-to-speech PersonaPlex model, not a text "
            "chat causal-LM checkpoint."
        )
        sys.exit(1)

    try:
        tokenizer = load_tokenizer(args.model_id)
        template_override = resolve_chat_template(args.chat_template, args.model_id, config_path=args._config_path)
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
        print("Tokenizer has no chat template. This TUI requires a chat model/tokenizer.")
        sys.exit(1)

    return TuiRuntime(model=model, tokenizer=tokenizer, args=args, input_device=input_device)


def main():
    args = parse_args()
    runtime = load_runtime(args)
    app = TuiChatApp(runtime)
    app.run()


if __name__ == "__main__":
    main()
