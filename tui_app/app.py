from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from dataclasses import dataclass

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.events import MouseScrollDown, MouseScrollUp
from textual.reactive import reactive
from textual.widgets import Input, Static

from tui_app.backends.base import BackendSession, Event
from tui_app.events import AnswerDelta, Error, Finish, Meta, ThinkDelta, TurnStart


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
    session: BackendSession
    args: argparse.Namespace


class UnifiedTuiApp(App):
    CSS = """
    Screen { layout: vertical; }
    #transcript { height: 1fr; padding: 1 1; layout: vertical; }
    .user-message { margin: 0 0 1 0; height: auto; }
    .assistant-message { margin: 0 0 1 0; layout: vertical; height: auto; }
    .thinking-panel { border: none; margin: 0 0 1 0; layout: vertical; height: auto; }
    #thinking-header { color: grey; text-style: bold; }
    #thinking-body { color: grey; margin: 0 0 0 2; height: auto; width: 100%; text-wrap: wrap; }
    .assistant-answer { color: white; height: auto; width: 100%; text-wrap: wrap; }
    .assistant-hint { margin: 1 0 0 0; }
    #input-band { dock: bottom; height: 3; background: #3a3a3a; padding: 0 1; }
    #chat-input { width: 100%; background: #3a3a3a; color: white; border: none; }
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
        self.messages: list[dict[str, str]] = []
        if runtime.args.system:
            self.messages.append({"role": "system", "content": runtime.args.system})

        self.event_queue: queue.Queue = queue.Queue()
        self.pending_assistant: AssistantMessage | None = None
        self.pending_turn_id = 0
        self.is_generating = False
        self.turn_records = []
        self.follow_output = True
        self._scroll_end_scheduled = False
        self._max_events_per_tick = max(20, int(runtime.args.ui_max_events_per_tick))

    def compose(self) -> ComposeResult:
        self.transcript = TranscriptPane(id="transcript")
        yield self.transcript
        with Container(id="input-band"):
            yield Input(placeholder="Type message and press Enter", id="chat-input")

    def on_mount(self):
        self.transcript.can_focus = True
        interval_s = max(0.01, float(self.runtime.args.ui_tick_ms) / 1000.0)
        self.set_interval(interval_s, self._drain_events)
        self.call_after_refresh(self._scroll_to_end_now)

    def action_toggle_latest_thinking(self):
        if self.pending_assistant is not None:
            self.pending_assistant.toggle_thinking()

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

        thread = threading.Thread(target=self._run_generation, args=(turn_id, list(self.messages)), daemon=True)
        thread.start()

    def _emit_event(self, ev: Event):
        self.event_queue.put(ev)

    def _run_generation(self, turn_id: int, messages: list[dict[str, str]]):
        self.runtime.session.generate_turn(turn_id=turn_id, messages=messages, emit=self._emit_event)

    def _drain_events(self):
        processed = 0
        pending_think_token_inc = 0
        while processed < self._max_events_per_tick:
            try:
                ev = self.event_queue.get_nowait()
            except queue.Empty:
                break
            processed += 1

            if isinstance(ev, TurnStart):
                continue

            if getattr(ev, "turn_id", None) != self.pending_turn_id or self.pending_assistant is None:
                continue

            if isinstance(ev, Meta):
                if ev.key == "think_tokens_inc":
                    pending_think_token_inc += int(ev.value)
                continue

            if pending_think_token_inc:
                self.pending_assistant.add_think_tokens(pending_think_token_inc)
                pending_think_token_inc = 0

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
                self.pending_assistant.finish(ended_in_think=False)
                self.is_generating = False
            elif isinstance(ev, Finish):
                record = ev.record
                self.pending_assistant.finish(ended_in_think=record.ended_in_think)
                if record.trimmed_messages is not None:
                    self.messages = list(record.trimmed_messages)
                assistant_for_history = record.answer if record.answer else record.think
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
        line = json.dumps(record.__dict__, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
