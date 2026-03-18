from __future__ import annotations

import argparse
import difflib
import json
import os
import queue
import shlex
import threading
import time
from dataclasses import dataclass
from typing import Callable

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.events import MouseScrollDown, MouseScrollUp
from textual.reactive import reactive
from textual.widgets import Static, TextArea

from tui_app.backends.base import BackendSession, Event
from tui_app.events import AnswerDelta, Error, Finish, Meta, ThinkDelta, TurnStart
from tui_app.knobs import build_intent_knobs
from tui_app.telemetry import TelemetryContext, build_runtime_sample_payload
from tui_app.tools import build_tool_runtime


@dataclass
class SlashCommand:
    name: str
    summary: str
    usage: str
    handler: Callable[["UnifiedTuiApp", list[str]], str]
    aliases: tuple[str, ...] = ()
    read_only: bool = True
    examples: tuple[str, ...] = ()
    hidden: bool = False


@dataclass
class ShowOptions:
    verbose: bool = False
    as_json: bool = False


@dataclass
class ShowTopic:
    name: str
    summary: str
    usage: str
    handler: Callable[["UnifiedTuiApp", list[str]], str]
    aliases: tuple[str, ...] = ()


class SlashRegistry:
    def __init__(self):
        self._commands: dict[str, SlashCommand] = {}
        self._aliases: dict[str, str] = {}

    def register(self, command: SlashCommand) -> None:
        key = command.name.lower()
        self._commands[key] = command
        for alias in command.aliases:
            self._aliases[alias.lower()] = key

    def resolve(self, name: str) -> SlashCommand | None:
        if not name:
            return None
        key = name.lower()
        if key in self._commands:
            return self._commands[key]
        target = self._aliases.get(key)
        if target:
            return self._commands.get(target)
        return None

    def canonical_names(self) -> list[str]:
        return sorted(self._commands.keys())

    def all_names(self) -> list[str]:
        names = set(self._commands.keys())
        names.update(self._aliases.keys())
        return sorted(names)

    def all_commands(self) -> list[SlashCommand]:
        return [self._commands[name] for name in self.canonical_names()]


def _format_name_grid(items: list[str], *, cols: int = 4, indent: str = "  ", min_width: int = 14) -> list[str]:
    if not items:
        return [f"{indent}(none)"]
    width = max(min_width, max(len(item) for item in items) + 2)
    rows: list[str] = []
    for start in range(0, len(items), cols):
        chunk = items[start : start + cols]
        rows.append(indent + "".join(item.ljust(width) for item in chunk).rstrip())
    return rows


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


class UserMessage(Static):
    def __init__(self, text: str, images: list[str] | None = None):
        super().__init__(classes="user-message")
        self.text = text
        self.images = images or []

    def on_mount(self):
        body = Text(f"You: {self.text}", style=SOFT_TEXT)
        if self.images:
            names = ", ".join(os.path.basename(p) for p in self.images[:6])
            suffix = "" if len(self.images) <= 6 else f" (+{len(self.images) - 6} more)"
            body.append(f"\nImages: {names}{suffix}", style="grey70")
        self.update(body)


class InfoMessage(Static):
    def __init__(self, command: str, text: str):
        super().__init__(classes="info-message")
        self.command = command
        self.text = text

    def on_mount(self):
        body = Text(f"# {self.command}\n", style="bright_black")
        body.append(self.text, style="grey70")
        self.update(body)


class ThinkingPanel(Container):
    expanded = reactive(False)
    running = reactive(False)
    phase = reactive(0)

    def __init__(self, start_expanded: bool, animate: bool):
        super().__init__(classes="thinking-panel")
        self.expanded = start_expanded
        self.animate = animate
        self.thinking_text = ""
        self.generated_tokens = 0
        self.started_at: float | None = None
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
        base = "model..."
        if self.started_at is None and self.total_seconds <= 0:
            suffix = " (waiting)"
        else:
            elapsed = (time.time() - self.started_at) if (self.running and self.started_at is not None) else self.total_seconds
            suffix = f" ({self.generated_tokens} tokens"
            if self.running:
                live_tps = (self.generated_tokens / elapsed) if elapsed > 0 else 0.0
                suffix += f", {elapsed:.1f}s, {live_tps:.1f} tok/sec"
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
            text.stylize(SOFT_TEXT, start, end)

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

    def start(self):
        if self.started_at is None:
            self.started_at = time.time()
        self.running = True
        self._render_header()

    def add_generated_tokens(self, token_inc: int):
        if token_inc <= 0:
            return
        self.generated_tokens += token_inc
        self._render_header()

    def append_thinking(self, text: str):
        if not text:
            return
        self.thinking_text += text
        self._render_header()
        if self.expanded:
            self._render_body()
        self.refresh(layout=True)

    def finish(self, *, ended_in_think: bool, completion_tokens: int | None = None, tokens_per_s: float | None = None):
        self.running = False
        if self.started_at is not None:
            self.total_seconds = max(0.0, time.time() - self.started_at)
        if completion_tokens is not None and completion_tokens >= 0:
            self.generated_tokens = int(completion_tokens)
        if tokens_per_s is not None:
            self.tokens_per_sec = float(tokens_per_s)
        else:
            self.tokens_per_sec = self.generated_tokens / self.total_seconds if self.total_seconds > 0 else 0.0
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
    def __init__(
        self,
        start_expanded: bool,
        animate_thinking: bool,
        show_tool_activity: bool,
        show_tool_arguments: bool,
    ):
        super().__init__(classes="assistant-message")
        self.start_expanded = start_expanded
        self.animate_thinking = animate_thinking
        self.show_tool_activity = show_tool_activity
        self.show_tool_arguments = show_tool_arguments
        self.answer_text = ""

    def compose(self) -> ComposeResult:
        self.thinking_panel = ThinkingPanel(start_expanded=self.start_expanded, animate=self.animate_thinking)
        self.answer_widget = Static(classes="assistant-answer")
        self.tools_widget = Static(classes="assistant-tools")
        self.hint_widget = Static(classes="assistant-hint")
        yield self.thinking_panel
        yield self.answer_widget
        yield self.tools_widget
        yield self.hint_widget

    def start_turn(self):
        self.thinking_panel.start()

    def add_generated_tokens(self, token_inc: int):
        self.thinking_panel.add_generated_tokens(token_inc=token_inc)

    def append_think(self, text: str):
        self.thinking_panel.append_thinking(text)

    def append_answer(self, text: str):
        if not text:
            return
        self.answer_text += text
        self.answer_widget.styles.height = "auto"
        self.answer_widget.update(Text(self.answer_text, style=SOFT_TEXT))
        self.refresh(layout=True)

    def finish(self, record):
        token_counts = dict(record.token_counts or {})
        throughput = dict(record.throughput or {})
        completion_tokens = token_counts.get("completion_tokens")
        tokens_per_s = throughput.get("tokens_per_s")
        self.thinking_panel.finish(
            ended_in_think=record.ended_in_think,
            completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
            tokens_per_s=float(tokens_per_s) if isinstance(tokens_per_s, (int, float)) else None,
        )
        self.tools_widget.update(
            self._render_tool_activity(
                record.tool_activity or [],
                show_activity=self.show_tool_activity,
                show_arguments=self.show_tool_arguments,
            )
        )
        if not self.answer_text.strip():
            msg = "(No final answer produced; generation ended during thinking. Increase max_new_tokens or adjust prompt.)"
            self.hint_widget.update(Text(msg, style="yellow"))

    def toggle_thinking(self):
        self.thinking_panel.toggle()

    @staticmethod
    def _render_tool_activity(
        tool_activity: list[dict[str, object]],
        *,
        show_activity: bool,
        show_arguments: bool,
    ) -> Text:
        if not show_activity or not tool_activity:
            return Text("")
        body = Text()
        for index, item in enumerate(tool_activity, start=1):
            if index > 1:
                body.append("\n\n")
            status = str(item.get("status", "") or "")
            name = str(item.get("name", "") or "")
            tool_call_id = str(item.get("tool_call_id", "") or "")
            body.append(f"[tool] {name}", style="bold cyan")
            if tool_call_id:
                body.append(f" id={tool_call_id}", style="grey70")
            if status:
                body.append(f" status={status}", style="yellow")
            if show_arguments:
                body.append("\nargs.raw:\n", style="grey70")
                body.append(str(item.get("arguments_raw", "") or ""), style=SOFT_TEXT)
                parsed = item.get("arguments_json")
                if parsed is not None:
                    body.append("\nargs.json:\n", style="grey70")
                    body.append(json.dumps(parsed, indent=2, ensure_ascii=False, default=str), style=SOFT_TEXT)
            result = item.get("result")
            error = item.get("error")
            if result not in (None, ""):
                body.append("\nresult:\n", style="grey70")
                body.append(str(result), style=SOFT_TEXT)
            if error not in (None, "") and error != result:
                body.append("\nerror:\n", style="grey70")
                body.append(str(error), style=SOFT_TEXT)
        return body


class TranscriptPane(VerticalScroll):
    def _line_step(self) -> int:
        try:
            step = int(getattr(self.app.runtime.args, "scroll_lines", 1))
        except Exception:
            step = 1
        return max(1, step)

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
        step = self._line_step()
        target = max(0.0, float(self.scroll_y) - step)
        self.scroll_to(y=target, animate=False)

    def action_page_up(self):
        self._break_follow()
        return super().action_page_up()

    def action_scroll_home(self):
        self._break_follow()
        return super().action_scroll_home()

    def action_scroll_down(self):
        step = self._line_step()
        target = min(float(self.max_scroll_y), float(self.scroll_y) + step)
        self.scroll_to(y=target, animate=False)
        if self._at_bottom():
            self._resume_follow()

    def action_page_down(self):
        result = super().action_page_down()
        if self._at_bottom():
            self._resume_follow()
        return result

    def action_scroll_end(self):
        result = super().action_scroll_end()
        # Keep End-key scroll behavior animated by not forcing an immediate non-animated scroll_end.
        self._resume_follow(immediate=False)
        return result

    def on_mouse_scroll_up(self, event: MouseScrollUp):
        self.action_scroll_up()
        event.stop()

    def on_mouse_scroll_down(self, event: MouseScrollDown):
        self.action_scroll_down()
        event.stop()


@dataclass
class TuiRuntime:
    session: BackendSession
    args: argparse.Namespace
    telemetry: TelemetryContext | None = None


class UnifiedTuiApp(App):
    CSS = """
    Screen { layout: vertical; }
    #transcript { height: 1fr; padding: 1 1; layout: vertical; }
    .user-message { margin: 0 0 1 0; height: auto; }
    .info-message { margin: 0 0 1 0; height: auto; width: 100%; text-wrap: wrap; color: grey; }
    .assistant-message { margin: 0 0 1 0; layout: vertical; height: auto; }
    .thinking-panel { border: none; margin: 0 0 1 0; layout: vertical; height: auto; }
    #thinking-header { color: grey; text-style: bold; }
    #thinking-body { color: grey; margin: 0 0 0 2; height: auto; width: 100%; text-wrap: wrap; }
    .assistant-answer { color: #e6dfcf; height: auto; width: 100%; text-wrap: wrap; }
    .assistant-tools { color: cyan; margin: 1 0 0 2; height: auto; width: 100%; text-wrap: wrap; }
    .assistant-hint { margin: 1 0 0 0; }
    #input-band { dock: bottom; height: 6; background: #3a3a3a; padding: 0 1; }
    #chat-input { width: 100%; height: 100%; background: #3a3a3a; color: #e6dfcf; border: none; }
    """

    BINDINGS = [
        Binding("t", "toggle_latest_thinking", "Toggle thinking"),
        Binding("ctrl+q", "exit_tui_only", "Exit TUI", priority=True),
        Binding("ctrl+x", "interrupt_or_quit_hint", "Interrupt generation", priority=True),
        Binding("enter", "submit_prompt", "Send", priority=True),
        Binding("shift+enter", "insert_newline", "New line", priority=True),
        Binding("ctrl+j", "insert_newline", "New line", priority=True),
        Binding("pageup", "scroll_page_up", "Scroll up", priority=True),
        Binding("pagedown", "scroll_page_down", "Scroll down", priority=True),
        Binding("home", "scroll_home", "Scroll top", priority=True),
        Binding("end", "scroll_end_manual", "Scroll bottom", priority=True),
    ]

    def __init__(self, runtime: TuiRuntime):
        super().__init__()
        self.runtime = runtime
        self.messages: list[dict[str, object]] = []
        if runtime.args.system:
            self.messages.append({"role": "system", "content": runtime.args.system})
        self.pending_images: list[str] = []
        self.pending_text_files: list[str] = []

        self.event_queue: queue.Queue = queue.Queue()
        self.pending_assistant: AssistantMessage | None = None
        self.pending_turn_id = 0
        self.is_generating = False
        self.generation_thread: threading.Thread | None = None
        self.turn_records = []
        self.follow_output = True
        self._scroll_end_scheduled = False
        self._max_events_per_tick = max(20, int(runtime.args.ui_max_events_per_tick))
        self.registry = SlashRegistry()
        self.show_topics: dict[str, ShowTopic] = {}
        self.show_aliases: dict[str, str] = {}
        self._active_show_opts: ShowOptions | None = None
        self.shutdown_backend_on_exit = True
        self._register_show_topics()
        self._register_commands()
        self._telemetry_finished = False
        self._telemetry_status = "finished"
        self._pending_turn_started_at: float | None = None
        self._pending_turn_first_token_at: float | None = None

    def compose(self) -> ComposeResult:
        self.transcript = TranscriptPane(id="transcript")
        yield self.transcript
        with Container(id="input-band"):
            yield TextArea(
                text="",
                soft_wrap=True,
                tab_behavior="focus",
                show_line_numbers=False,
                placeholder="Type message. Enter sends. New line: Ctrl+J (Shift+Enter if terminal supports it).",
                id="chat-input",
            )

    def on_mount(self):
        self.transcript.can_focus = True
        interval_s = max(0.01, float(self.runtime.args.ui_tick_ms) / 1000.0)
        self.set_interval(interval_s, self._drain_events)
        telemetry = self.runtime.telemetry
        sample_interval = float(getattr(self.runtime.args, "telemetry_sample_interval_s", 0.0) or 0.0)
        if telemetry is not None and telemetry.enabled and sample_interval > 0:
            self.set_interval(max(0.25, sample_interval), self._emit_runtime_sample)
        self.call_after_refresh(self._scroll_to_end_now)
        # Keep typing flow immediate: start with the input focused.
        self.query_one("#chat-input", TextArea).focus()

    def on_unmount(self):
        telemetry = self.runtime.telemetry
        if telemetry is not None and telemetry.enabled and not self._telemetry_finished:
            telemetry.publish_session_finished(status=self._telemetry_status)
            self._telemetry_finished = True

    def _emit_runtime_sample(self) -> None:
        telemetry = self.runtime.telemetry
        if telemetry is None or not telemetry.enabled:
            return
        generated_tokens = 0
        elapsed_s = 0.0
        if self.is_generating and self.pending_assistant is not None:
            generated_tokens = int(getattr(self.pending_assistant.thinking_panel, "generated_tokens", 0) or 0)
            started_at = getattr(self.pending_assistant.thinking_panel, "started_at", None)
            if isinstance(started_at, (int, float)):
                elapsed_s = max(0.0, time.time() - float(started_at))
        payload = build_runtime_sample_payload(
            telemetry=telemetry,
            session=self.runtime.session,
            generated_tokens=generated_tokens,
            elapsed_s=elapsed_s,
            requests_completed_total=len(self.turn_records),
            requests_in_flight=1 if self.is_generating else 0,
        )
        telemetry.publish_runtime_sample(payload)

    def action_toggle_latest_thinking(self):
        if self.pending_assistant is not None:
            self.pending_assistant.toggle_thinking()

    def action_exit_tui_only(self):
        self.shutdown_backend_on_exit = False
        self.exit()

    def action_interrupt_or_quit_hint(self):
        if not self.is_generating:
            self.notify("Use Ctrl+Q to quit.", severity="warning")
            return
        if self.pending_assistant is None:
            self.is_generating = False
            self._pending_turn_started_at = None
            self._pending_turn_first_token_at = None
            self.pending_turn_id += 1
            self.notify("Generation stopped.", severity="information")
            return

        self.pending_assistant.append_answer("\n[Generation stopped by user]")
        self.pending_assistant.thinking_panel.finish(ended_in_think=False)
        self.is_generating = False
        self._pending_turn_started_at = None
        self._pending_turn_first_token_at = None
        # Advance turn id so late events from the interrupted worker are ignored.
        self.pending_turn_id += 1
        if self._should_autofollow():
            self._request_scroll_end()
        self.notify("Generation stopped.", severity="information")

    def _has_active_generation_worker(self) -> bool:
        if self.generation_thread is None:
            return False
        if self.generation_thread.is_alive():
            return True
        self.generation_thread = None
        return False

    def action_insert_newline(self):
        input_box = self.query_one("#chat-input", TextArea)
        input_box.insert("\n")

    def _break_follow(self):
        self.follow_output = False
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

    async def action_submit_prompt(self):
        input_box = self.query_one("#chat-input", TextArea)
        text = input_box.text.strip()
        if not text and not self.pending_images:
            if not self.pending_text_files:
                input_box.load_text("")
                return
        if not text and not self.pending_images and not self.pending_text_files:
            input_box.load_text("")
            return
        if text.startswith("//"):
            text = text[1:]
        elif text.startswith("/"):
            await self._run_slash_command(text)
            input_box.load_text("")
            return
        if self.is_generating:
            self.notify("Generation in progress. Wait for current turn to finish.")
            return
        if self._has_active_generation_worker():
            self.notify("Previous generation is still shutting down. Please wait a moment.")
            return
        if text.lower() in {"exit", "quit"}:
            self.shutdown_backend_on_exit = text.lower() == "quit"
            self.exit()
            return
        if text.lower() == "clear":
            self.transcript.remove_children()
            self.messages = []
            self.turn_records = []
            self.pending_images.clear()
            self.pending_text_files.clear()
            self.pending_assistant = None
            if self.runtime.args.system:
                self.messages.append({"role": "system", "content": self.runtime.args.system})
            input_box.load_text("")
            return

        if text:
            user_text = f"{self.runtime.args.user_prefix}{text}" if self.runtime.args.user_prefix else text
        else:
            user_text = ""

        if self.pending_text_files:
            attached_blocks = []
            for path in self.pending_text_files:
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        content = fh.read()
                except Exception as exc:
                    self.notify(f"Failed to read attached file: {path} ({exc})", severity="error")
                    continue
                attached_blocks.append(f"[Attached file: {os.path.basename(path)}]\n{content}")
            self.pending_text_files.clear()
            if attached_blocks:
                file_text = "\n\n".join(attached_blocks)
                user_text = f"{user_text}\n\n{file_text}".strip() if user_text else file_text

        message: dict[str, object] = {"role": "user", "content": user_text}
        images = None
        if self.pending_images:
            images = list(self.pending_images)
            message["images"] = images
            self.pending_images.clear()
        self.messages.append(message)
        await self.transcript.mount(UserMessage(text or "(image)", images=images))

        assistant = AssistantMessage(
            start_expanded=self.runtime.args.show_thinking,
            animate_thinking=not self.runtime.args.no_animate_thinking,
            show_tool_activity=bool(getattr(self.runtime.args, "show_tool_activity", False)),
            show_tool_arguments=bool(getattr(self.runtime.args, "show_tool_arguments", False)),
        )
        await self.transcript.mount(assistant)
        self.pending_assistant = assistant
        self.pending_turn_id += 1
        turn_id = self.pending_turn_id
        self.is_generating = True
        self._pending_turn_started_at = None
        self._pending_turn_first_token_at = None

        self.follow_output = True
        self.transcript.refresh(layout=True)
        self._request_scroll_end()
        input_box.load_text("")

        thread = threading.Thread(target=self._run_generation, args=(turn_id, list(self.messages)), daemon=True)
        self.generation_thread = thread
        thread.start()

    async def _append_info(self, command: str, output: str):
        if not self.is_running:
            return
        transcript = getattr(self, "transcript", None)
        if transcript is None or not getattr(transcript, "is_attached", False):
            return
        await self.transcript.mount(InfoMessage(command=command, text=output))
        if self._should_autofollow():
            self._request_scroll_end()

    async def _run_slash_command(self, raw_text: str):
        cmd_text = raw_text if raw_text.startswith("/") else f"/{raw_text}"
        if cmd_text == "/":
            cmd_text = "/help"
        try:
            tokens = shlex.split(cmd_text)
        except ValueError:
            await self._append_info(
                raw_text,
                "Error: Unterminated quote. Tip: wrap strings in matching quotes or escape them.",
            )
            return
        if not tokens:
            tokens = ["/help"]

        name = tokens[0][1:].strip().lower()
        argv = tokens[1:]
        command = self.registry.resolve(name)
        if not command:
            candidates = self.registry.all_names()
            matches = difflib.get_close_matches(name, candidates, n=3, cutoff=0.5)
            msg = f"Unknown command: /{name}"
            if matches:
                msg += "\nDid you mean: " + ", ".join(f"/{m}" for m in matches)
            await self._append_info(raw_text, msg)
            return
        if self.is_generating and not command.read_only:
            await self._append_info(raw_text, "Not allowed during generation.")
            return

        try:
            output = command.handler(self, argv).strip()
        except Exception as exc:
            output = f"Command failed: {exc}"
        if command.name == "exit":
            return
        if not output:
            output = "(No output)"
        await self._append_info(raw_text, output)

    def _show_opts(self) -> ShowOptions:
        return self._active_show_opts or ShowOptions()

    def _to_json_or_lines(self, data: dict, lines: list[str]) -> str:
        if self._show_opts().as_json:
            return json.dumps(data, indent=2, ensure_ascii=False, default=str)
        return "\n".join(lines)

    def _parse_show_flags(self, argv: list[str]) -> tuple[list[str], ShowOptions, str | None]:
        opts = ShowOptions()
        rest: list[str] = []
        idx = 0
        while idx < len(argv):
            token = argv[idx]
            if token in {"--verbose", "-v"}:
                opts.verbose = True
            elif token == "--json":
                opts.as_json = True
            elif token in {"--help", "-h", "?"}:
                return [], opts, "help"
            else:
                rest.append(token)
            idx += 1
        return rest, opts, None

    def _register_show_topics(self):
        topics = [
            ShowTopic("status", "High-level current state", "/show status", UnifiedTuiApp._show_status),
            ShowTopic("session", "Session overview", "/show session", UnifiedTuiApp._show_session),
            ShowTopic("think", "Thinking and history-strip settings", "/show think", UnifiedTuiApp._show_think),
            ShowTopic("prompt", "System/prefix/template settings", "/show prompt", UnifiedTuiApp._show_prompt),
            ShowTopic("gen", "Effective generation settings", "/show gen", UnifiedTuiApp._show_gen),
            ShowTopic("tools", "Tool harness and backend tool support", "/show tools", UnifiedTuiApp._show_tools),
            ShowTopic("recording", "Request capture and transcript saving", "/show recording", UnifiedTuiApp._show_recording),
            ShowTopic("connection", "Backend connection/attach state", "/show connection", UnifiedTuiApp._show_connection),
            ShowTopic("backend", "Backend details", "/show backend", UnifiedTuiApp._show_backend),
            ShowTopic("last", "Last turn record summary", "/show last", UnifiedTuiApp._show_last),
            ShowTopic("history", "Conversation history summary", "/show history", UnifiedTuiApp._show_history),
            ShowTopic("request", "Last captured request payload (sanitized)", "/show request", UnifiedTuiApp._show_request),
            ShowTopic("logs", "Recent backend logs", "/show logs [--n N] [--filter TEXT]", UnifiedTuiApp._show_logs),
            ShowTopic("files", "Resolved file paths", "/show files", UnifiedTuiApp._show_files),
            ShowTopic("config", "Loaded config layers and origins", "/show config", UnifiedTuiApp._show_config),
            ShowTopic("model", "Model/backend identifiers", "/show model", UnifiedTuiApp._show_model),
            ShowTopic("ui", "UI behavior settings", "/show ui", UnifiedTuiApp._show_ui),
            ShowTopic("env", "Environment summary", "/show env", UnifiedTuiApp._show_env),
            ShowTopic("args", "Parsed CLI args", "/show args", UnifiedTuiApp._show_args),
            ShowTopic("aliases", "Alias map for slash commands/topics", "/show aliases", UnifiedTuiApp._show_aliases),
        ]
        self.show_topics = {topic.name: topic for topic in topics}
        self.show_aliases = {
            "session": "session",
            "status": "status",
            "think": "think",
            "prompt": "prompt",
            "gen": "gen",
            "recording": "recording",
            "connection": "connection",
            "ui": "ui",
            "args": "args",
            "history": "history",
            "last": "last",
            "env": "env",
            "files": "files",
            "model": "model",
            "config": "config",
            "backend": "backend",
            "tools": "tools",
            "aliases": "aliases",
            "logs": "logs",
            "request": "request",
        }

    def _register_commands(self):
        self.registry.register(
            SlashCommand(
                name="help",
                aliases=("?",),
                summary="List commands or show command help",
                usage="/help [command]",
                handler=UnifiedTuiApp._cmd_help,
                examples=("/help", "/help show"),
            )
        )
        self.registry.register(
            SlashCommand(
                name="show",
                summary="Inspect runtime, prompt, backend, and turn state",
                usage="/show <topic>",
                handler=UnifiedTuiApp._cmd_show,
                examples=("/show", "/show status", "/show think", "/show connection"),
            )
        )
        self.registry.register(
            SlashCommand(
                name="status",
                summary="Concise runtime summary",
                usage="/status",
                handler=UnifiedTuiApp._cmd_status,
            )
        )
        self.registry.register(
            SlashCommand(
                name="system",
                summary="Show active system prompt and source",
                usage="/system",
                handler=UnifiedTuiApp._cmd_system,
            )
        )
        self.registry.register(
            SlashCommand(
                name="prefix",
                summary="Show prefix/prompt-mode values and applicability",
                usage="/prefix",
                handler=UnifiedTuiApp._cmd_prefix,
            )
        )
        self.registry.register(
            SlashCommand(
                name="toolblocks",
                aliases=("tool-blocks",),
                summary="Toggle tool block visibility in the transcript",
                usage="/toolblocks [on|off|toggle]",
                handler=UnifiedTuiApp._cmd_toolblocks,
                read_only=False,
            )
        )
        self.registry.register(
            SlashCommand(
                name="toolargs",
                aliases=("tool-args",),
                summary="Toggle tool argument visibility in transcript tool blocks",
                usage="/toolargs [on|off|toggle]",
                handler=UnifiedTuiApp._cmd_toolargs,
                read_only=False,
            )
        )
        self.registry.register(
            SlashCommand(
                name="image",
                aliases=("img",),
                summary="Attach an image to the next prompt (HF vision models only)",
                usage="/image <path> | /image list | /image clear",
                handler=UnifiedTuiApp._cmd_image,
                examples=("/image ./pic.png", "/img list", "/image clear"),
            )
        )
        self.registry.register(
            SlashCommand(
                name="file",
                aliases=("attach", "textfile"),
                summary="Attach a text file to the next prompt",
                usage="/file <path> | /file list | /file clear",
                handler=UnifiedTuiApp._cmd_file,
                examples=("/file ./notes.txt", "/file list", "/file clear"),
            )
        )
        for alias_name in (
            "model",
            "config",
            "env",
            "files",
            "session",
            "think",
            "prompt",
            "gen",
            "recording",
            "connection",
            "ui",
            "args",
            "history",
            "last",
            "backend",
            "tools",
            "logs",
            "request",
        ):
            self.registry.register(
                SlashCommand(
                    name=alias_name,
                    summary=f"Alias for /show {alias_name}",
                    usage=f"/{alias_name}",
                    handler=lambda app, argv, topic=alias_name: app._cmd_show([topic, *argv]),
                    hidden=True,
                )
            )
        self.registry.register(
            SlashCommand(
                name="clear",
                summary="Clear transcript and chat history",
                usage="/clear",
                handler=UnifiedTuiApp._cmd_clear,
                read_only=False,
            )
        )
        self.registry.register(
            SlashCommand(
                name="exit",
                summary="Exit TUI and keep managed backend running",
                usage="/exit",
                handler=UnifiedTuiApp._cmd_exit,
            )
        )
        self.registry.register(
            SlashCommand(
                name="quit",
                summary="Exit TUI and shut down managed backend",
                usage="/quit",
                handler=UnifiedTuiApp._cmd_quit,
            )
        )

    def _cmd_help(self, argv: list[str]) -> str:
        if not argv:
            lines = [
                "TUI commands",
                "",
                "Inspect",
            ]
            lines.extend(_format_name_grid(["/show", "/status", "/help <command>"], cols=3, min_width=22))
            lines.extend(
                [
                    "",
                    "Prompt + Session",
                ]
            )
            lines.extend(_format_name_grid(["/system", "/prefix", "/image", "/file"], cols=4, min_width=18))
            lines.extend(
                [
                    "",
                    "Transcript View",
                ]
            )
            lines.extend(_format_name_grid(["/toolblocks", "/toolargs"], cols=3, min_width=18))
            lines.extend(
                [
                    "",
                    "Session Control",
                ]
            )
            lines.extend(_format_name_grid(["/clear", "/exit", "/quit"], cols=3, min_width=18))
            lines.extend(
                [
                    "",
                    "Start Here",
                ]
            )
            lines.extend(_format_name_grid(["/show", "/show status", "/show session", "/show think", "/show connection"], cols=3, min_width=22))
            lines.extend(
                [
                    "",
                    "Use /help --all to include hidden aliases.",
                ]
            )
            return "\n".join(lines)
        if argv[0] in {"--all", "aliases"}:
            lines = ["All commands (including hidden aliases):"]
            for cmd in self.registry.all_commands():
                hidden = " [hidden]" if cmd.hidden else ""
                lines.append(f"/{cmd.name}{hidden}: {cmd.summary}")
            return "\n".join(lines)

        target = argv[0].lstrip("/").lower()
        cmd = self.registry.resolve(target)
        if not cmd:
            candidates = self.registry.all_names()
            matches = difflib.get_close_matches(target, candidates, n=3, cutoff=0.5)
            msg = f"Unknown command: /{target}"
            if matches:
                msg += "\nDid you mean: " + ", ".join(f"/{m}" for m in matches)
            return msg
        lines = [
            f"/{cmd.name}",
            f"Summary: {cmd.summary}",
            f"Usage: {cmd.usage}",
        ]
        if cmd.aliases:
            lines.append("Aliases: " + ", ".join(f"/{a}" for a in cmd.aliases))
        if cmd.examples:
            lines.append("Examples:")
            for ex in cmd.examples:
                lines.append(f"  {ex}")
        return "\n".join(lines)

    def _cmd_show(self, argv: list[str]) -> str:
        if not argv:
            lines = [
                "Usage: /show <topic> [--verbose] [--json]",
                "",
                "Overview",
            ]
            lines.extend(_format_name_grid(["status", "session"], cols=4))
            lines.extend(
                [
                    "",
                    "Prompting",
                ]
            )
            lines.extend(_format_name_grid(["think", "prompt", "gen", "tools"], cols=4))
            lines.extend(
                [
                    "",
                    "Backend + Debug",
                ]
            )
            lines.extend(_format_name_grid(["connection", "backend", "last", "history", "request", "logs"], cols=4))
            lines.extend(
                [
                    "",
                    "Files + Config",
                ]
            )
            lines.extend(_format_name_grid(["files", "config", "model", "ui", "env", "args", "aliases"], cols=4))
            lines.extend(
                [
                    "",
                    "Examples:",
                    "  /show status",
                    "  /show think",
                    "  /show backend --verbose",
                    "  /show last --verbose",
                ]
            )
            return "\n".join(lines)
        topic = argv[0].lower()
        parsed_argv, show_opts, parse_err = self._parse_show_flags(argv[1:])
        if topic in {"?", "--help"}:
            return self._cmd_show([])
        topic_key = self.show_aliases.get(topic, topic)
        handler = self.show_topics.get(topic_key)
        if not handler:
            matches = difflib.get_close_matches(topic, sorted(self.show_topics.keys()), n=3, cutoff=0.5)
            msg = f"Unknown show topic: {topic}"
            if matches:
                msg += "\nDid you mean: " + ", ".join(matches)
            return msg
        if parse_err == "help":
            return f"{handler.usage}\n{handler.summary}"
        if parse_err:
            return parse_err
        if parsed_argv and parsed_argv[0] in {"?", "--help"}:
            return f"{handler.usage}\n{handler.summary}"
        if show_opts.as_json and topic_key == "logs":
            return "Topic 'logs' does not support --json."
        self._active_show_opts = show_opts
        try:
            return handler.handler(self, parsed_argv)
        finally:
            self._active_show_opts = None

    def _cmd_status(self, argv: list[str]) -> str:
        del argv
        self._active_show_opts = ShowOptions(verbose=False, as_json=False)
        try:
            return self._show_status([])
        finally:
            self._active_show_opts = None

    def _cmd_system(self, argv: list[str]) -> str:
        del argv
        args = self.runtime.args
        system = args.system or ""
        cli_overrides = set(getattr(args, "_cli_overrides", set()) or set())
        source = "none"
        if "system" in cli_overrides:
            source = "--system"
        elif args.system_file:
            source = "--system-file"
        elif system:
            source = "config/default"
        lines = [f"source: {source}", f"has_system: {bool(system)}"]
        if args.system_file:
            resolved = resolve_path_maybe_relative(args.system_file, config_path=args._config_path)
            lines.append(f"system_file: {resolved}")
        lines.append("system_prompt:")
        lines.append(system if system else "(empty)")
        return "\n".join(lines)

    def _cmd_prefix(self, argv: list[str]) -> str:
        del argv
        args = self.runtime.args
        backend = self.runtime.session.backend_name
        prompt_mode = getattr(args, "prompt_mode", None) if backend == "hf" else None
        chat_template = getattr(args, "chat_template", None)
        lines = [
            f"backend: {backend}",
            f"user_prefix: {args.user_prefix!r}",
            "prompt_prefix: N/A",
            f"prompt_mode: {prompt_mode if prompt_mode is not None else 'N/A'}",
            f"chat_template: {chat_template if chat_template else 'N/A'}",
            f"history_strip_think: {bool(getattr(args, 'history_strip_think', False))}",
        ]
        return "\n".join(lines)

    def _cmd_image(self, argv: list[str]) -> str:
        backend = self.runtime.session.backend_name
        if backend not in {"hf", "openai", "vllm"}:
            return f"Images are only supported on HF and OpenAI-compatible backends (current backend={backend})."

        if not argv:
            return "\n".join(
                [
                    "Usage:",
                    "  /image <path>        Attach an image file to the next user message",
                    "  /image list          Show pending image attachments",
                    "  /image clear         Clear pending image attachments",
                ]
            )

        sub = argv[0].strip().lower()
        if sub in {"list", "ls"}:
            if not self.pending_images:
                return "(No pending images.)"
            lines = ["Pending images:"]
            for idx, path in enumerate(self.pending_images, start=1):
                lines.append(f"{idx}. {path}")
            return "\n".join(lines)

        if sub in {"clear", "reset"}:
            count = len(self.pending_images)
            self.pending_images.clear()
            return f"Cleared {count} pending image(s)."

        raw_path = " ".join(argv).strip()
        path = resolve_path_maybe_relative(raw_path, config_path=self.runtime.args._config_path)
        if not os.path.isfile(path):
            return f"Not found: {path}"
        self.pending_images.append(path)
        return f"Attached: {path}\n(Will be sent with the next message.)"

    def _cmd_file(self, argv: list[str]) -> str:
        if not argv:
            return "\n".join(
                [
                    "Usage:",
                    "  /file <path>         Attach a text file to the next user message",
                    "  /file list           Show pending text-file attachments",
                    "  /file clear          Clear pending text-file attachments",
                ]
            )

        sub = argv[0].strip().lower()
        if sub in {"list", "ls"}:
            if not self.pending_text_files:
                return "(No pending text files.)"
            lines = ["Pending text files:"]
            for idx, path in enumerate(self.pending_text_files, start=1):
                lines.append(f"{idx}. {path}")
            return "\n".join(lines)

        if sub in {"clear", "reset"}:
            count = len(self.pending_text_files)
            self.pending_text_files.clear()
            return f"Cleared {count} pending text file(s)."

        raw_path = " ".join(argv).strip()
        path = resolve_path_maybe_relative(raw_path, config_path=self.runtime.args._config_path)
        if not os.path.isfile(path):
            return f"Not found: {path}"
        try:
            with open(path, "rb") as fh:
                sample = fh.read(4096)
        except Exception as exc:
            return f"Failed to read: {path} ({exc})"
        if b"\x00" in sample:
            return f"Refusing to attach binary file: {path}"
        self.pending_text_files.append(path)
        return f"Attached text file: {path}\n(Will be sent with the next message.)"

    def _cmd_toolargs(self, argv: list[str]) -> str:
        current = bool(getattr(self.runtime.args, "show_tool_arguments", False))
        if not argv:
            return f"show_tool_arguments: {current}"
        token = argv[0].strip().lower()
        if token in {"on", "true", "1"}:
            current = True
        elif token in {"off", "false", "0"}:
            current = False
        elif token in {"toggle", "flip"}:
            current = not current
        else:
            return "Usage: /toolargs [on|off|toggle]"
        self.runtime.args.show_tool_arguments = current
        return f"show_tool_arguments: {current}"

    def _cmd_toolblocks(self, argv: list[str]) -> str:
        current = bool(getattr(self.runtime.args, "show_tool_activity", False))
        if not argv:
            return f"show_tool_activity: {current}"
        token = argv[0].strip().lower()
        if token in {"on", "true", "1"}:
            current = True
        elif token in {"off", "false", "0"}:
            current = False
        elif token in {"toggle", "flip"}:
            current = not current
        else:
            return "Usage: /toolblocks [on|off|toggle]"
        self.runtime.args.show_tool_activity = current
        return f"show_tool_activity: {current}"

    def _cmd_clear(self, argv: list[str]) -> str:
        del argv
        self.transcript.remove_children()
        self.messages = []
        self.turn_records = []
        self.pending_images.clear()
        self.pending_text_files.clear()
        if self.runtime.args.system:
            self.messages.append({"role": "system", "content": self.runtime.args.system})
        self.pending_assistant = None
        return "Transcript, conversation history, turn records, and pending attachments cleared."

    def _cmd_exit(self, argv: list[str]) -> str:
        del argv
        self.shutdown_backend_on_exit = False
        self.exit()
        return "Exiting."

    def _cmd_quit(self, argv: list[str]) -> str:
        del argv
        self.shutdown_backend_on_exit = True
        self.exit()
        return "Exiting and shutting down backend."

    def _backend_summary_line(self) -> str:
        session = self.runtime.session
        describe = getattr(session, "describe", None)
        if not callable(describe):
            return "(unavailable)"
        info = describe()
        if not isinstance(info, dict):
            return str(info)
        backend = session.backend_name
        if backend == "vllm":
            return (
                f"managed={info.get('managed_mode')} pid={info.get('pid')} "
                f"base_url={info.get('base_url')} model={info.get('model_id')}"
            )
        if backend == "openai":
            return f"base_url={info.get('base_url')} model={info.get('model_id')}"
        if backend == "ollama":
            return f"model={session.resolved_model_id}"
        return ", ".join(f"{k}={info[k]}" for k in sorted(info.keys()))

    @staticmethod
    def _token_counts_view(record) -> dict[str, object]:
        counts = dict(record.token_counts or {})
        return {
            "prompt_tokens": counts.get("prompt_tokens", "unavailable"),
            "completion_tokens": counts.get("completion_tokens", "unavailable"),
            "total_tokens": counts.get("total_tokens", "unavailable"),
        }

    @staticmethod
    def _tokens_per_s_view(record) -> object:
        throughput = dict(record.throughput or {})
        return throughput.get("tokens_per_s", "unavailable")

    @staticmethod
    def _context_view(record) -> dict[str, object]:
        context = dict(record.context or {})
        return {
            "strategy": context.get("strategy", "unavailable"),
            "dropped_messages": context.get("dropped_messages", 0),
            "fit": context.get("fit", "unavailable"),
            "system_message_preserved": context.get("system_message_preserved", "unavailable"),
            "system_drop_required": context.get("system_drop_required", "unavailable"),
            "reserved_generation_tokens": context.get("reserved_generation_tokens", "unavailable"),
            "context_window": context.get("context_window", "unavailable"),
        }

    def _show_status(self, argv: list[str]) -> str:
        del argv
        args = self.runtime.args
        session = self.runtime.session
        profile = getattr(args, "_config_profile", "") or ""
        system_text = args.system or ""
        data = {
            "backend": session.backend_name,
            "model": session.resolved_model_id,
            "config_path": args._config_path or "(none)",
            "profile": profile or "(none)",
            "generating": bool(self.is_generating),
            "follow_output": bool(self.follow_output),
            "pending_images": len(self.pending_images),
            "prompt": {
                "has_system": bool(system_text),
                "history_strip_think": bool(getattr(args, "history_strip_think", False)),
                "tools_enabled": bool(getattr(args, "tools_enabled", False)),
            },
            "generation": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
            "backend_summary": self._backend_summary_line(),
        }
        logs_fn = getattr(session, "get_recent_logs", None)
        if callable(logs_fn):
            try:
                data["log_tail_available"] = len(logs_fn(1)) > 0
            except Exception:
                data["log_tail_available"] = False
        if self.turn_records:
            last = self.turn_records[-1]
            token_counts = self._token_counts_view(last)
            context = self._context_view(last)
            data["last_turn"] = {
                "elapsed_s": last.timing.get("elapsed"),
                "ended_in_think": bool(last.ended_in_think),
                "chars_raw": len(last.raw),
                "chars_think": len(last.think),
                "chars_answer": len(last.answer),
                "tool_activity_count": len(last.tool_activity or []),
                "prompt_tokens": token_counts["prompt_tokens"],
                "completion_tokens": token_counts["completion_tokens"],
                "total_tokens": token_counts["total_tokens"],
                "tokens_per_s": self._tokens_per_s_view(last),
                "context": context,
            }
        else:
            data["last_turn"] = "(none)"
        lines = [
            f"backend: {data['backend']}",
            f"model: {data['model']}",
            f"config: {data['config_path']} (profile={data['profile']})",
            (
                "state: "
                f"generating={str(data['generating']).lower()} "
                f"follow_output={str(data['follow_output']).lower()} "
                f"pending_images={data['pending_images']}"
            ),
            (
                "prompt: "
                f"system={data['prompt']['has_system']} "
                f"history_strip={data['prompt']['history_strip_think']} "
                f"tools={data['prompt']['tools_enabled']}"
            ),
            (
                "generation: "
                f"max_new_tokens={data['generation']['max_new_tokens']} "
                f"temperature={data['generation']['temperature']} "
                f"top_p={data['generation']['top_p']}"
            ),
            f"connection: {data['backend_summary']}",
        ]
        if isinstance(data["last_turn"], dict):
            lt = data["last_turn"]
            lines.extend(
                [
                    "last_turn:",
                    f"  elapsed_s={lt['elapsed_s']} ended_in_think={lt['ended_in_think']} tools={lt['tool_activity_count']}",
                    (
                        "  tokens: "
                        f"prompt={lt['prompt_tokens']} completion={lt['completion_tokens']} "
                        f"total={lt['total_tokens']} tok/s={lt['tokens_per_s']}"
                    ),
                    (
                        "  context: "
                        f"strategy={lt['context']['strategy']} "
                        f"dropped={lt['context']['dropped_messages']} "
                        f"system_preserved={lt['context']['system_message_preserved']}"
                    ),
                ]
            )
            if self._show_opts().verbose:
                lines.extend(
                    [
                        (
                            "  chars: "
                            f"raw={lt['chars_raw']} think={lt['chars_think']} answer={lt['chars_answer']}"
                        ),
                        f"  log_tail_available: {data.get('log_tail_available', False)}",
                    ]
                )
        else:
            lines.append("last_turn: (none)")
        return self._to_json_or_lines(data, lines)

    def _show_think(self, argv: list[str]) -> str:
        del argv
        args = self.runtime.args
        data = {
            "assume_think": bool(getattr(args, "assume_think", False)),
            "show_thinking": bool(getattr(args, "show_thinking", False)),
            "history_strip_think": bool(getattr(args, "history_strip_think", False)),
            "no_animate_thinking": bool(getattr(args, "no_animate_thinking", False)),
        }
        lines = [
            f"assume_think: {data['assume_think']}",
            f"show_thinking: {data['show_thinking']}",
            f"history_strip_think: {data['history_strip_think']}",
        ]
        if self._show_opts().verbose:
            lines.append(f"no_animate_thinking: {data['no_animate_thinking']}")
        return self._to_json_or_lines(data, lines)

    def _show_session(self, argv: list[str]) -> str:
        del argv
        args = self.runtime.args
        session = self.runtime.session
        profile = getattr(args, "_config_profile", "") or ""
        system_text = args.system or ""
        system_source = "none"
        cli_overrides = set(getattr(args, "_cli_overrides", set()) or set())
        if "system" in cli_overrides:
            system_source = "--system"
        elif args.system_file:
            system_source = "--system-file"
        elif system_text:
            system_source = "config/default"
        describe = getattr(session, "describe", None)
        backend_info = describe() if callable(describe) else {}
        if not isinstance(backend_info, dict):
            backend_info = {"summary": str(backend_info)}
        data = {
            "backend": session.backend_name,
            "model_id": session.resolved_model_id,
            "config_path": args._config_path or "(none)",
            "profile": profile or "(none)",
            "state": {
                "is_generating": bool(self.is_generating),
                "follow_output": bool(self.follow_output),
            },
            "prompt": {
                "has_system": bool(system_text),
                "system_source": system_source,
                "system_file": resolve_path_maybe_relative(args.system_file, config_path=args._config_path)
                if args.system_file
                else "(none)",
                "user_prefix": args.user_prefix or "",
                "chat_template": args.chat_template or "(none)",
            },
            "thinking": {
                "assume_think": bool(getattr(args, "assume_think", False)),
                "show_thinking": bool(getattr(args, "show_thinking", False)),
                "history_strip_think": bool(getattr(args, "history_strip_think", False)),
            },
            "tools": {
                "enabled": bool(getattr(args, "tools_enabled", False)),
                "mode": getattr(args, "tools_mode", "off"),
                "show_tool_activity": bool(getattr(args, "show_tool_activity", False)),
                "show_tool_arguments": bool(getattr(args, "show_tool_arguments", False)),
                "max_calls_per_turn": getattr(args, "tools_max_calls_per_turn", "unavailable"),
            },
            "recording": {
                "capture_last_request": bool(getattr(args, "capture_last_request", False)),
                "save_transcript": (
                    resolve_path_maybe_relative(args.save_transcript, config_path=args._config_path)
                    if getattr(args, "save_transcript", "")
                    else "(none)"
                ),
            },
            "backend_connection": backend_info,
        }
        lines = [
            f"backend: {data['backend']}",
            f"model_id: {data['model_id']}",
            f"config: {data['config_path']} (profile={data['profile']})",
            f"state: generating={data['state']['is_generating']} follow_output={data['state']['follow_output']}",
            (
                "prompt: "
                f"system={data['prompt']['has_system']} "
                f"source={data['prompt']['system_source']} "
                f"user_prefix={data['prompt']['user_prefix']!r}"
            ),
            (
                "thinking: "
                f"assume={data['thinking']['assume_think']} "
                f"show={data['thinking']['show_thinking']} "
                f"history_strip={data['thinking']['history_strip_think']}"
            ),
            (
                "tools: "
                f"enabled={data['tools']['enabled']} "
                f"mode={data['tools']['mode']} "
                f"show_blocks={data['tools']['show_tool_activity']} "
                f"show_args={data['tools']['show_tool_arguments']}"
            ),
            (
                "recording: "
                f"capture_last_request={data['recording']['capture_last_request']} "
                f"save_transcript={data['recording']['save_transcript']}"
            ),
        ]
        backend_summary = self._backend_summary_line()
        if backend_summary and backend_summary != "(unavailable)":
            lines.append(f"connection: {backend_summary}")
        if self._show_opts().verbose:
            lines.extend(
                [
                    f"prompt.chat_template: {data['prompt']['chat_template']}",
                    f"prompt.system_file: {data['prompt']['system_file']}",
                    f"tools.max_calls_per_turn: {data['tools']['max_calls_per_turn']}",
                    f"pending_images: {len(self.pending_images)}",
                    f"transcript_widgets: {len(self.transcript.children)}",
                    f"scroll_y: {float(self.transcript.scroll_y):.1f}/{float(self.transcript.max_scroll_y):.1f}",
                ]
            )
            for key in sorted(backend_info.keys()):
                lines.append(f"backend.{key}: {backend_info[key]}")
        return self._to_json_or_lines(data, lines)

    def _extract_vllm_runtime_flags(self) -> dict[str, object]:
        raw = list(getattr(self.runtime.args, "vllm_extra_args", []) or [])
        out: dict[str, object] = {
            "kv_cache_dtype": "none",
            "calculate_kv_scales": False,
            "stream_interval": "default",
            "swap_space": "default",
            "max_num_seqs": "default",
        }
        idx = 0
        while idx < len(raw):
            token = raw[idx]
            nxt = raw[idx + 1] if idx + 1 < len(raw) else None
            if token == "--kv-cache-dtype" and nxt is not None:
                out["kv_cache_dtype"] = nxt
                idx += 2
                continue
            if token == "--calculate-kv-scales":
                out["calculate_kv_scales"] = True
                idx += 1
                continue
            if token == "--stream-interval" and nxt is not None:
                out["stream_interval"] = nxt
                idx += 2
                continue
            if token == "--swap-space" and nxt is not None:
                out["swap_space"] = nxt
                idx += 2
                continue
            if token == "--max-num-seqs" and nxt is not None:
                out["max_num_seqs"] = nxt
                idx += 2
                continue
            idx += 1
        return out

    def _backend_runtime_summary(self) -> dict[str, object]:
        args = self.runtime.args
        backend = self.runtime.session.backend_name
        summary: dict[str, object] = {"backend": backend}
        if backend == "hf":
            weights_quant = "8bit" if getattr(args, "use_8bit", False) else "4bit" if getattr(args, "use_4bit", False) else "none"
            summary.update(
                {
                    "weights_quantization": weights_quant,
                    "kv_cache_quantization": "none",
                    "attention_backend": args.hf_attn_implementation or "default",
                    "device_map": getattr(args, "hf_device_map", "") or "auto",
                    "max_memory": getattr(args, "hf_max_memory", "") or "(unset)",
                    "low_cpu_mem_usage": getattr(args, "hf_low_cpu_mem_usage", None),
                    "text_only_mode": bool(getattr(args, "hf_text_only", False)),
                    "supports_images": bool(getattr(self.runtime.session, "supports_images", False)),
                    "context_window_target": args.max_context_tokens if args.max_context_tokens is not None else "model_default",
                    "context_allocation": "grows_with_sequence",
                }
            )
            return summary
        if backend == "vllm":
            flags = self._extract_vllm_runtime_flags()
            summary.update(
                {
                    "weights_quantization": "none",
                    "kv_cache_quantization": flags["kv_cache_dtype"],
                    "calculate_kv_scales": bool(flags["calculate_kv_scales"]),
                    "attention_backend": args.vllm_attention_backend or "auto",
                    "context_window_target": int(args.vllm_max_model_len or 0) or "server_default",
                    "context_allocation": "preallocated_kv_cache",
                    "stream_interval": flags["stream_interval"],
                    "max_num_seqs": flags["max_num_seqs"],
                    "swap_space": flags["swap_space"],
                }
            )
            return summary
        if backend == "gguf":
            summary.update(
                {
                    "weights_quantization": "gguf",
                    "kv_cache_quantization": "none",
                    "context_window_target": args.n_ctx,
                    "context_allocation": "reserved_context_window",
                }
            )
            return summary
        if backend == "exl2":
            summary.update(
                {
                    "weights_quantization": "exl2",
                    "kv_cache_quantization": getattr(args, "cache_type", "fp16"),
                    "context_window_target": getattr(args, "max_seq_len", None) or "model_default",
                    "context_allocation": "reserved_context_window",
                }
            )
            return summary
        if backend in {"openai", "ollama"}:
            summary.update(
                {
                    "weights_quantization": "server_managed",
                    "kv_cache_quantization": "server_managed",
                    "context_window_target": "server_managed",
                    "context_allocation": "server_managed",
                }
            )
            return summary
        return summary

    def _backend_runtime_views(self) -> tuple[dict[str, object], dict[str, object]]:
        args = self.runtime.args
        session = self.runtime.session
        backend = session.backend_name
        requested: dict[str, object] = {}
        effective: dict[str, object] = {}
        describe = getattr(session, "describe", None)
        info = describe() if callable(describe) else {}
        if not isinstance(info, dict):
            info = {}

        if backend == "hf":
            weights_quant = "8bit" if bool(getattr(args, "use_8bit", False)) else "4bit" if bool(getattr(args, "use_4bit", False)) else "none"
            requested = {
                "weights_quantization": weights_quant,
                "kv_cache_quantization": "none",
                "attention_backend": getattr(args, "hf_attn_implementation", None) or "default",
                "device_map": getattr(args, "hf_device_map", "") or "auto",
                "max_memory": getattr(args, "hf_max_memory", "") or "(unset)",
                "low_cpu_mem_usage": getattr(args, "hf_low_cpu_mem_usage", None),
                "context_window_target": args.max_context_tokens if args.max_context_tokens is not None else "model_default",
                "text_only_mode": bool(getattr(args, "hf_text_only", False)),
            }
            effective = {
                "weights_quantization": info.get("weights_quantization", weights_quant),
                "kv_cache_quantization": info.get("kv_cache_quantization", "none"),
                "attention_backend": info.get("attention_backend_effective", "unknown"),
                "device_map": info.get("hf_device_map", requested["device_map"]),
                "max_memory": info.get("hf_max_memory", requested["max_memory"]),
                "max_memory_effective": info.get("hf_max_memory_effective", "unknown"),
                "low_cpu_mem_usage": info.get("hf_low_cpu_mem_usage", requested["low_cpu_mem_usage"]),
                "torch_dtype": info.get("torch_dtype_effective", info.get("torch_dtype", getattr(args, "dtype", "auto"))),
                "runtime_device": info.get("runtime_device", "unknown"),
                "fully_on_single_gpu": info.get("fully_on_single_gpu", "unknown"),
                "modules_on_cpu": info.get("modules_on_cpu", "unknown"),
                "modules_on_disk": info.get("modules_on_disk", "unknown"),
                "memory_footprint": info.get("memory_footprint", "unavailable"),
                "cuda_memory_allocated": info.get("cuda_memory_allocated", "unavailable"),
                "cuda_memory_reserved": info.get("cuda_memory_reserved", "unavailable"),
                "cuda_max_memory_allocated": info.get("cuda_max_memory_allocated", "unavailable"),
                "cuda_max_memory_reserved": info.get("cuda_max_memory_reserved", "unavailable"),
                "qwen_fast_path_available": info.get("qwen_fast_path_available", "unknown"),
                "context_window_target": requested["context_window_target"],
                "context_allocation": info.get("context_allocation", "grows_with_sequence"),
                "text_only_mode": info.get("text_only_mode", requested["text_only_mode"]),
                "supports_images": info.get("supports_images", False),
            }
            return requested, effective

        if backend == "vllm":
            flags = self._extract_vllm_runtime_flags()
            requested = {
                "weights_quantization": "none",
                "kv_cache_quantization": flags["kv_cache_dtype"],
                "calculate_kv_scales": bool(flags["calculate_kv_scales"]),
                "attention_backend": getattr(args, "vllm_attention_backend", "") or "auto",
                "context_window_target": int(args.vllm_max_model_len or 0) or "server_default",
                "stream_interval": flags["stream_interval"],
                "max_num_seqs": flags["max_num_seqs"],
                "swap_space": flags["swap_space"],
            }
            effective = {
                "weights_quantization": "none",
                "kv_cache_quantization": requested["kv_cache_quantization"],
                "calculate_kv_scales": requested["calculate_kv_scales"],
                "attention_backend": info.get("attention_backend_effective", requested["attention_backend"]),
                "context_window_target": requested["context_window_target"],
                "context_allocation": "preallocated_kv_cache",
                "stream_interval": requested["stream_interval"],
                "max_num_seqs": requested["max_num_seqs"],
                "swap_space": requested["swap_space"],
            }
            return requested, effective

        if backend == "gguf":
            requested = {
                "weights_quantization": "gguf",
                "kv_cache_quantization": "none",
                "context_window_target": args.n_ctx,
            }
            effective = {
                **requested,
                "context_allocation": "reserved_context_window",
            }
            return requested, effective

        if backend == "exl2":
            requested = {
                "weights_quantization": "exl2",
                "kv_cache_quantization": getattr(args, "cache_type", "fp16"),
                "context_window_target": getattr(args, "max_seq_len", None) or "model_default",
            }
            effective = {
                **requested,
                "context_allocation": "reserved_context_window",
            }
            return requested, effective

        if backend in {"openai", "ollama"}:
            requested = {
                "weights_quantization": "server_managed",
                "kv_cache_quantization": "server_managed",
                "context_window_target": "server_managed",
            }
            effective = {
                **requested,
                "context_allocation": "server_managed",
            }
            return requested, effective

        return requested, effective

    def _show_prompt(self, argv: list[str]) -> str:
        del argv
        args = self.runtime.args
        system = args.system or ""
        cli_overrides = set(getattr(args, "_cli_overrides", set()) or set())
        source = "none"
        if "system" in cli_overrides:
            source = "--system"
        elif args.system_file:
            source = "--system-file"
        elif system:
            source = "config/default"
        data = {
            "has_system": bool(system),
            "system_source": source,
            "system_file": resolve_path_maybe_relative(args.system_file, config_path=args._config_path)
            if args.system_file
            else "(none)",
            "user_prefix": args.user_prefix or "",
            "prompt_mode": getattr(args, "prompt_mode", None) if self.runtime.session.backend_name == "hf" else "N/A",
            "chat_template": args.chat_template or "(none)",
        }
        lines = [
            f"has_system: {data['has_system']}",
            f"system_source: {data['system_source']}",
            f"user_prefix: {data['user_prefix']!r}",
            f"chat_template: {data['chat_template']}",
        ]
        if self._show_opts().verbose:
            lines.append(f"system_file: {data['system_file']}")
            lines.append(f"prompt_mode: {data['prompt_mode']}")
            lines.append("system_prompt:")
            lines.append(system if system else "(empty)")
        return self._to_json_or_lines(data, lines)

    def _show_gen(self, argv: list[str]) -> str:
        del argv
        args = self.runtime.args
        if self.turn_records:
            data = dict(self.turn_records[-1].knobs or {})
        else:
            data = build_intent_knobs(args, self.runtime.session.backend_name)

        sent = dict(data.get("sent") or {})
        deferred = list(data.get("deferred") or [])
        ignored = list(data.get("ignored") or [])
        notes = list(data.get("notes") or [])

        lines = []
        mode = data.get("mode")
        if mode:
            lines.append(f"mode: {mode}")
        lines.append("sent.*")
        if sent:
            lines.extend([f"  {k}: {v!r}" for k, v in sent.items()])
        else:
            lines.append("  (none)")
        if self._show_opts().verbose:
            lines.append("deferred")
            if deferred:
                lines.extend([f"  {k}" for k in deferred])
            else:
                lines.append("  (none)")
            lines.append("ignored")
            if ignored:
                lines.extend([f"  {k}" for k in ignored])
            else:
                lines.append("  (none)")
            if notes:
                lines.append("notes")
                lines.extend([f"  {row}" for row in notes])
        else:
            lines.append(f"deferred: {deferred if deferred else []}")
            lines.append(f"ignored: {ignored if ignored else []}")

        if self._show_opts().verbose:
            data["args"] = {
                key: ("***" if key.endswith("api_key") or key == "api_key" else value)
                for key, value in vars(args).items()
                if not key.startswith("_")
            }
        return self._to_json_or_lines(data, lines)

    def _show_ui(self, argv: list[str]) -> str:
        del argv
        args = self.runtime.args
        data = {
            "show_thinking": args.show_thinking,
            "show_tool_activity": bool(getattr(args, "show_tool_activity", False)),
            "show_tool_arguments": bool(getattr(args, "show_tool_arguments", False)),
            "no_animate_thinking": args.no_animate_thinking,
            "scroll_lines": args.scroll_lines,
            "ui_tick_ms": args.ui_tick_ms,
            "ui_max_events_per_tick": args.ui_max_events_per_tick,
            "follow_output": self.follow_output,
            "capture_last_request": bool(getattr(args, "capture_last_request", False)),
        }
        lines = [f"{k}: {v}" for k, v in data.items()]
        return self._to_json_or_lines(data, lines)

    def _show_recording(self, argv: list[str]) -> str:
        del argv
        args = self.runtime.args
        data = {
            "capture_last_request": bool(getattr(args, "capture_last_request", False)),
            "save_transcript": (
                resolve_path_maybe_relative(args.save_transcript, config_path=args._config_path)
                if getattr(args, "save_transcript", "")
                else "(none)"
            ),
            "telemetry_jsonl": (
                resolve_path_maybe_relative(args.telemetry_jsonl, config_path=args._config_path)
                if getattr(args, "telemetry_jsonl", "")
                else "(none)"
            ),
            "telemetry_sample_interval_s": float(getattr(args, "telemetry_sample_interval_s", 0.0) or 0.0),
        }
        lines = [
            f"capture_last_request: {data['capture_last_request']}",
            f"save_transcript: {data['save_transcript']}",
            f"telemetry_jsonl: {data['telemetry_jsonl']}",
            f"telemetry_sample_interval_s: {data['telemetry_sample_interval_s']}",
        ]
        return self._to_json_or_lines(data, lines)

    def _show_args(self, argv: list[str]) -> str:
        del argv
        data = {}
        for key, value in vars(self.runtime.args).items():
            if key.startswith("_"):
                continue
            if key.endswith("api_key") or key == "api_key":
                data[key] = "***" if value else value
            else:
                data[key] = value
        lines = [f"{k}: {data[k]!r}" for k in sorted(data.keys())]
        return self._to_json_or_lines(data, lines)

    def _show_history(self, argv: list[str]) -> str:
        del argv
        roles = [msg.get("role", "?") for msg in self.messages[-10:]]
        data = {
            "turn_records": len(self.turn_records),
            "messages": len(self.messages),
            "last_roles_10": roles,
        }
        lines = [f"{k}: {v}" for k, v in data.items()]
        return self._to_json_or_lines(data, lines)

    def _show_last(self, argv: list[str]) -> str:
        del argv
        if not self.turn_records:
            return "No completed turns yet."
        last = self.turn_records[-1]
        token_counts = self._token_counts_view(last)
        context = self._context_view(last)
        data = {
            "backend": last.backend,
            "model_id": last.model_id,
            "ended_in_think": last.ended_in_think,
            "elapsed": last.timing.get("elapsed"),
            "chars_raw": len(last.raw),
            "chars_think": len(last.think),
            "chars_answer": len(last.answer),
            "prompt_tokens": token_counts["prompt_tokens"],
            "completion_tokens": token_counts["completion_tokens"],
            "total_tokens": token_counts["total_tokens"],
            "tokens_per_s": self._tokens_per_s_view(last),
            "context": context,
            "tool_activity_count": len(last.tool_activity or []),
        }
        finish_reason = None
        if isinstance(last.gen, dict):
            finish_reason = last.gen.get("finish_reason")
        if finish_reason is not None:
            data["finish_reason"] = finish_reason
        data["len_raw"] = data["chars_raw"]
        data["len_think"] = data["chars_think"]
        data["len_answer"] = data["chars_answer"]
        lines = [f"{k}: {v}" for k, v in data.items()]
        if self._show_opts().verbose:
            data["raw"] = last.raw
            data["think"] = last.think
            data["answer"] = last.answer
            data["tool_activity"] = list(last.tool_activity or [])
            lines.extend(["", "raw:", last.raw, "", "think:", last.think, "", "answer:", last.answer])
            lines.extend(["", "tool_activity:"])
            if last.tool_activity:
                for index, item in enumerate(last.tool_activity, start=1):
                    lines.append(f"  [{index}] name={item.get('name', '')} id={item.get('tool_call_id', '')}")
                    lines.append(f"    status: {item.get('status', '')}")
                    lines.append(f"    arguments_raw: {item.get('arguments_raw', '')}")
                    parsed = item.get("arguments_json")
                    if parsed is not None:
                        lines.append("    arguments_json:")
                        for row in json.dumps(parsed, indent=2, ensure_ascii=False, default=str).splitlines():
                            lines.append(f"      {row}")
                    if item.get("result") not in (None, ""):
                        lines.append(f"    result: {item.get('result')}")
                    if item.get("error") not in (None, ""):
                        lines.append(f"    error: {item.get('error')}")
            else:
                lines.append("  (none)")
        return self._to_json_or_lines(data, lines)

    def _show_env(self, argv: list[str]) -> str:
        del argv
        data = {
            "OLLAMA_HOST": os.environ.get("OLLAMA_HOST", "(unset)"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"),
            "OPENAI_API_KEY": "***" if os.environ.get("OPENAI_API_KEY") else "(unset)",
        }
        lines = [f"{k}: {v}" for k, v in data.items()]
        return self._to_json_or_lines(data, lines)

    def _show_files(self, argv: list[str]) -> str:
        del argv
        args = self.runtime.args
        data = {"config_path": args._config_path or "(none)"}
        lines = [f"config_path: {data['config_path']}"]
        if args.system_file:
            data["system_file"] = resolve_path_maybe_relative(args.system_file, config_path=args._config_path)
            lines.append(f"system_file: {data['system_file']}")
        else:
            data["system_file"] = "(none)"
            lines.append("system_file: (none)")
        if args.save_transcript:
            data["save_transcript"] = resolve_path_maybe_relative(args.save_transcript, config_path=args._config_path)
            lines.append(f"save_transcript: {data['save_transcript']}")
        else:
            data["save_transcript"] = "(none)"
            lines.append("save_transcript: (none)")
        return self._to_json_or_lines(data, lines)

    def _show_model(self, argv: list[str]) -> str:
        del argv
        data = {
            "backend": self.runtime.session.backend_name,
            "model_id": self.runtime.session.resolved_model_id,
        }
        lines = [f"backend: {data['backend']}", f"model_id: {data['model_id']}"]
        return self._to_json_or_lines(data, lines)

    def _show_config(self, argv: list[str]) -> str:
        del argv
        args = self.runtime.args
        data = {"config_path": args._config_path or "(none)"}
        lines = [f"config_path: {data['config_path']}"]
        profile = getattr(args, "_config_profile", "") or ""
        data["profile"] = profile or "(none)"
        lines.append(f"profile: {data['profile']}")
        loaded = list(getattr(args, "_config_layers", []) or [])
        data["loaded_files"] = loaded
        if loaded:
            lines.append("loaded_files:")
            for path in loaded:
                lines.append(f"  - {path}")
        else:
            lines.append("loaded_files: (none)")
        origins = dict(getattr(args, "_config_origins", {}) or {})
        data["origins"] = origins
        lines.append("origins:")
        if origins:
            for key in sorted(origins.keys()):
                lines.append(f"  {key}: {origins[key]}")
        else:
            lines.append("  (none)")
        return self._to_json_or_lines(data, lines)

    def _show_backend(self, argv: list[str]) -> str:
        del argv
        session = self.runtime.session
        requested_runtime, effective_runtime = self._backend_runtime_views()
        data = {
            "backend": session.backend_name,
            "resolved_model_id": session.resolved_model_id,
            "effective": effective_runtime,
        }
        lines = [
            f"backend: {data['backend']}",
            f"resolved_model_id: {data['resolved_model_id']}",
        ]
        preferred = [
            "weights_quantization",
            "kv_cache_quantization",
            "calculate_kv_scales",
            "torch_dtype",
            "attention_backend",
            "runtime_device",
            "device_map",
            "fully_on_single_gpu",
            "modules_on_cpu",
            "modules_on_disk",
            "memory_footprint",
            "cuda_memory_allocated",
            "cuda_memory_reserved",
            "max_memory_effective",
            "qwen_fast_path_available",
            "max_memory",
            "low_cpu_mem_usage",
            "text_only_mode",
            "supports_images",
            "context_window_target",
            "context_allocation",
            "stream_interval",
            "max_num_seqs",
            "swap_space",
        ]
        for key in preferred:
            if key in effective_runtime:
                lines.append(f"{key}: {effective_runtime[key]}")
        describe = getattr(session, "describe", None)
        info = getattr(session, "get_info", None)
        extra = None
        if callable(describe):
            extra = describe()
        elif callable(info):
            extra = info()
        if isinstance(extra, dict):
            data["describe"] = extra
            if self._show_opts().verbose:
                data["requested"] = requested_runtime
                lines.append("requested:")
                if requested_runtime:
                    for key in sorted(requested_runtime.keys()):
                        lines.append(f"  {key}: {requested_runtime[key]}")
                else:
                    lines.append("  (none)")
                lines.append("effective:")
                if effective_runtime:
                    for key in sorted(effective_runtime.keys()):
                        lines.append(f"  {key}: {effective_runtime[key]}")
                else:
                    lines.append("  (none)")
                lines.append("raw:")
                for key in sorted(extra.keys()):
                    lines.append(f"  {key}: {extra[key]}")
        return self._to_json_or_lines(data, lines)

    def _show_connection(self, argv: list[str]) -> str:
        del argv
        session = self.runtime.session
        describe = getattr(session, "describe", None)
        info = describe() if callable(describe) else {}
        if not isinstance(info, dict):
            info = {}
        data = {
            "backend": session.backend_name,
            "model_id": session.resolved_model_id,
            "summary": self._backend_summary_line(),
            "backend_info": info,
        }
        lines = [
            f"backend: {data['backend']}",
            f"model_id: {data['model_id']}",
            f"summary: {data['summary']}",
        ]
        if self._show_opts().verbose:
            for key in sorted(info.keys()):
                lines.append(f"{key}: {info[key]}")
        return self._to_json_or_lines(data, lines)

    def _show_tools(self, argv: list[str]) -> str:
        del argv
        runtime = build_tool_runtime(self.runtime.args, self.runtime.session.backend_name)
        data = runtime.describe(verbose=self._show_opts().verbose)
        backend_tooling: dict[str, object] = {}
        if self.runtime.session.backend_name == "vllm":
            backend_tooling = {
                "enable_auto_tool_choice": self.runtime.args.vllm_enable_auto_tool_choice,
                "tool_call_parser": self.runtime.args.vllm_tool_call_parser or "",
                "tool_choice": getattr(self.runtime.args, "tools_tool_choice", "") or "",
            }
        elif self.runtime.session.backend_name == "openai":
            backend_tooling = {
                "tool_choice": getattr(self.runtime.args, "tools_tool_choice", "") or "",
            }
        data["backend_tooling"] = backend_tooling
        lines = [
            f"backend: {data['backend']}",
            f"supported_backend: {data['supported_backend']}",
            f"enabled: {data['enabled']}",
            f"mode: {data['mode']}",
            f"schema_file: {data['schema_file'] or '(none)'}",
            f"tool_choice: {data.get('tool_choice') or '(default)'}",
            f"allow: {data['allow']}",
            f"deny: {data['deny']}",
            f"max_calls_per_turn: {data['max_calls_per_turn']}",
            f"timeout_s: {data['timeout_s']}",
            f"max_result_chars: {data['max_result_chars']}",
            f"tool_names: {data['tool_names']}",
        ]
        if backend_tooling:
            lines.append("backend_tooling:")
            for key, value in backend_tooling.items():
                lines.append(f"  {key}: {value}")
        if self._show_opts().verbose:
            lines.append("available_tools:")
            for name, item in sorted((data.get("available_tools") or {}).items()):
                lines.append(f"  {name}: source={item['source']} executable={item['executable']}")
        return self._to_json_or_lines(data, lines)

    def _show_logs(self, argv: list[str]) -> str:
        n = 0
        filt = ""
        explicit_n = False
        explicit_filter = False
        idx = 0
        while idx < len(argv):
            token = argv[idx]
            if token == "--n":
                if idx + 1 >= len(argv):
                    return "Missing value for --n"
                try:
                    n = max(1, int(argv[idx + 1]))
                except ValueError:
                    return f"Invalid --n value: {argv[idx + 1]}"
                explicit_n = True
                idx += 2
                continue
            if token == "--filter":
                if idx + 1 >= len(argv):
                    return "Missing value for --filter"
                filt = argv[idx + 1]
                explicit_filter = True
                idx += 2
                continue
            return f"Unknown logs arg: {token}"

        effective_verbose = self._show_opts().verbose or explicit_n or explicit_filter
        if n <= 0:
            n = 80 if effective_verbose else 20

        session = self.runtime.session
        logs_fn = getattr(session, "get_recent_logs", None)
        if not callable(logs_fn):
            return f"Backend '{session.backend_name}' does not expose in-memory logs."
        rows = logs_fn(n, None)
        if filt:
            rows = [r for r in rows if filt in r]

        args = self.runtime.args
        key = f"{session.backend_name}_log_file"
        configured_log_file = getattr(args, key, "") if hasattr(args, key) else ""
        lines = [f"backend: {session.backend_name}", f"requested_n: {n}", f"filter: {filt or '(none)'}"]
        list_sources_fn = getattr(session, "list_log_sources", None)
        if callable(list_sources_fn):
            try:
                sources = list_sources_fn()
            except Exception:
                sources = []
            lines.append(f"available_sources: {', '.join(sources) if sources else '(none)'}")
        if configured_log_file:
            lines.append(f"log_file: {configured_log_file}")
        else:
            lines.append("log_file: (not configured)")
        if not effective_verbose and not rows:
            lines.append("No logs captured in backend ring buffer yet.")
            lines.append("Hint: run a prompt, then `/show logs` again or use `/show logs --n 80`.")
            return "\n".join(lines)
        if rows:
            lines.append("tail:")
            lines.extend(rows)
        else:
            lines.append("tail: (empty)")
        return "\n".join(lines)

    def _show_request(self, argv: list[str]) -> str:
        del argv
        session = self.runtime.session
        req_fn = getattr(session, "get_last_request", None)
        if not callable(req_fn):
            return (
                f"Backend '{session.backend_name}' does not expose request capture.\n"
                "No request captured (enable request capture in config: [ui] capture_last_request = true)."
            )
        payload = req_fn()
        if not payload:
            return "No request captured (enable request capture in config: [ui] capture_last_request = true)."
        if self._show_opts().as_json:
            return json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        lines = [
            f"backend: {session.backend_name}",
            f"captured: {bool(payload)}",
            f"keys: {sorted(payload.keys())}",
        ]
        if self._show_opts().verbose:
            lines.append("")
            lines.append(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return "\n".join(lines)

    def _show_aliases(self, argv: list[str]) -> str:
        del argv
        cmd_alias_map: dict[str, list[str]] = {}
        show_alias_map: dict[str, str] = {}
        shortcuts: dict[str, str] = {}
        lines = ["Command aliases:"]
        for name in sorted(self.registry.canonical_names()):
            cmd = self.registry.resolve(name)
            if cmd is None or not cmd.aliases:
                continue
            cmd_alias_map[name] = [f"/{alias}" for alias in cmd.aliases]
            lines.append(f"  /{name}: " + ", ".join(f"/{alias}" for alias in cmd.aliases))

        lines.append("")
        lines.append("Show-topic aliases:")
        for alias in sorted(self.show_aliases.keys()):
            target = self.show_aliases[alias]
            if alias == target:
                continue
            show_alias_map[alias] = target
            lines.append(f"  {alias} -> {target}")

        lines.append("")
        lines.append("Command shortcuts to /show <topic>:")
        for name in sorted(self.registry.canonical_names()):
            cmd = self.registry.resolve(name)
            if cmd is None:
                continue
            if cmd.summary.startswith("Alias for /show "):
                shortcuts[f"/{name}"] = cmd.summary.removeprefix("Alias for ")
                lines.append(f"  /{name} -> {cmd.summary.removeprefix('Alias for ')}")
        return self._to_json_or_lines(
            {"command_aliases": cmd_alias_map, "show_aliases": show_alias_map, "show_shortcuts": shortcuts},
            lines,
        )

    def _emit_event(self, ev: Event):
        self.event_queue.put(ev)

    def _run_generation(self, turn_id: int, messages: list[dict[str, str]]):
        self.runtime.session.generate_turn(turn_id=turn_id, messages=messages, emit=self._emit_event)

    def _drain_events(self):
        processed = 0
        pending_generated_token_inc = 0
        while processed < self._max_events_per_tick:
            try:
                ev = self.event_queue.get_nowait()
            except queue.Empty:
                break
            processed += 1

            if isinstance(ev, TurnStart):
                if getattr(ev, "turn_id", None) == self.pending_turn_id and self.pending_assistant is not None:
                    self._pending_turn_started_at = time.time()
                    self.pending_assistant.start_turn()
                continue

            if getattr(ev, "turn_id", None) != self.pending_turn_id or self.pending_assistant is None:
                continue

            if isinstance(ev, Meta):
                if ev.key == "generated_tokens_inc":
                    token_inc = int(ev.value)
                    if token_inc > 0 and self._pending_turn_first_token_at is None:
                        self._pending_turn_first_token_at = time.time()
                    pending_generated_token_inc += token_inc
                continue

            if pending_generated_token_inc:
                self.pending_assistant.add_generated_tokens(pending_generated_token_inc)
                pending_generated_token_inc = 0

            if isinstance(ev, ThinkDelta):
                self.pending_assistant.append_think(ev.text)
                if self._should_autofollow():
                    self._request_scroll_end()
            elif isinstance(ev, AnswerDelta):
                self.pending_assistant.append_answer(ev.text)
                if self._should_autofollow():
                    self._request_scroll_end()
            elif isinstance(ev, Error):
                self.pending_assistant.append_answer(f"\n[Generation error] {ev.message}")
                self.pending_assistant.thinking_panel.finish(ended_in_think=False)
                self.is_generating = False
                self._pending_turn_started_at = None
                self._pending_turn_first_token_at = None
                telemetry = self.runtime.telemetry
                if telemetry is not None and telemetry.enabled:
                    telemetry.publish_error(scope="turn", message=ev.message)
            elif isinstance(ev, Finish):
                record = ev.record
                self.pending_assistant.finish(record)
                if record.trimmed_messages is not None:
                    self.messages = list(record.trimmed_messages)
                else:
                    assistant_for_history = record.answer if record.answer else record.think
                    self.messages.append({"role": "assistant", "content": assistant_for_history})
                self.turn_records.append(record)
                if self.runtime.args.save_transcript:
                    self._append_transcript_record(record)
                telemetry = self.runtime.telemetry
                if telemetry is not None and telemetry.enabled:
                    timing = record.timing if isinstance(record.timing, dict) else {}
                    if "time_to_first_token" not in timing and self._pending_turn_first_token_at is not None:
                        started_at = timing.get("start")
                        if not isinstance(started_at, (int, float)):
                            started_at = self._pending_turn_started_at
                        if isinstance(started_at, (int, float)):
                            timing["time_to_first_token"] = max(
                                0.0,
                                float(self._pending_turn_first_token_at) - float(started_at),
                            )
                    telemetry.publish_turn_finished(turn_id=ev.turn_id, record=record)
                self.is_generating = False
                self._pending_turn_started_at = None
                self._pending_turn_first_token_at = None
                if self._should_autofollow():
                    self._request_scroll_end()

        if pending_generated_token_inc and self.pending_assistant is not None:
            self.pending_assistant.add_generated_tokens(pending_generated_token_inc)
        if processed > 0:
            self.transcript.refresh(layout=True)

    def _append_transcript_record(self, record):
        path = resolve_path_maybe_relative(self.runtime.args.save_transcript, config_path=self.runtime.args._config_path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        line = json.dumps(record.__dict__, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
SOFT_TEXT = "#e6dfcf"
