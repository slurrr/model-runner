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


@dataclass
class SlashCommand:
    name: str
    summary: str
    usage: str
    handler: Callable[["UnifiedTuiApp", list[str]], str]
    aliases: tuple[str, ...] = ()
    read_only: bool = True
    examples: tuple[str, ...] = ()
    include_in_help: bool = True


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
    .assistant-answer { color: white; height: auto; width: 100%; text-wrap: wrap; }
    .assistant-hint { margin: 1 0 0 0; }
    #input-band { dock: bottom; height: 6; background: #3a3a3a; padding: 0 1; }
    #chat-input { width: 100%; height: 100%; background: #3a3a3a; color: white; border: none; }
    """

    BINDINGS = [
        Binding("t", "toggle_latest_thinking", "Toggle thinking"),
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
        self.registry = SlashRegistry()
        self.show_topics: dict[str, ShowTopic] = {}
        self.show_aliases: dict[str, str] = {}
        self._register_show_topics()
        self._register_commands()

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
        self.call_after_refresh(self._scroll_to_end_now)
        # Keep typing flow immediate: start with the input focused.
        self.query_one("#chat-input", TextArea).focus()

    def action_toggle_latest_thinking(self):
        if self.pending_assistant is not None:
            self.pending_assistant.toggle_thinking()

    def action_interrupt_or_quit_hint(self):
        if not self.is_generating:
            self.notify("Use Ctrl+Q to quit.", severity="warning")
            return
        if self.pending_assistant is None:
            self.is_generating = False
            self.pending_turn_id += 1
            self.notify("Generation stopped.", severity="information")
            return

        self.pending_assistant.append_answer("\n[Generation stopped by user]")
        self.pending_assistant.finish(ended_in_think=False)
        self.is_generating = False
        # Advance turn id so late events from the interrupted worker are ignored.
        self.pending_turn_id += 1
        if self._should_autofollow():
            self._request_scroll_end()
        self.notify("Generation stopped.", severity="information")

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
        if not text:
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
        if text.lower() in {"exit", "quit"}:
            self.exit()
            return
        if text.lower() == "clear":
            self.transcript.remove_children()
            self.messages = []
            if self.runtime.args.system:
                self.messages.append({"role": "system", "content": self.runtime.args.system})
            input_box.load_text("")
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
        input_box.load_text("")

        thread = threading.Thread(target=self._run_generation, args=(turn_id, list(self.messages)), daemon=True)
        thread.start()

    async def _append_info(self, command: str, output: str):
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

    def _register_show_topics(self):
        topics = [
            ShowTopic("session", "Session/backend state", "/show session", UnifiedTuiApp._show_session),
            ShowTopic("prompt", "Prompt-related settings", "/show prompt", UnifiedTuiApp._show_prompt),
            ShowTopic("gen", "Effective generation settings", "/show gen", UnifiedTuiApp._show_gen),
            ShowTopic("ui", "UI behavior settings", "/show ui", UnifiedTuiApp._show_ui),
            ShowTopic("args", "Parsed CLI args", "/show args", UnifiedTuiApp._show_args),
            ShowTopic("history", "Conversation history summary", "/show history", UnifiedTuiApp._show_history),
            ShowTopic("last", "Last turn record summary", "/show last", UnifiedTuiApp._show_last),
            ShowTopic("env", "Environment summary", "/show env", UnifiedTuiApp._show_env),
            ShowTopic("files", "Resolved file paths", "/show files", UnifiedTuiApp._show_files),
            ShowTopic("model", "Model/backend identifiers", "/show model", UnifiedTuiApp._show_model),
            ShowTopic("config", "Loaded config path", "/show config", UnifiedTuiApp._show_config),
            ShowTopic("backend", "Backend details", "/show backend", UnifiedTuiApp._show_backend),
            ShowTopic("aliases", "Alias map for slash commands/topics", "/show aliases", UnifiedTuiApp._show_aliases),
        ]
        self.show_topics = {topic.name: topic for topic in topics}
        self.show_aliases = {
            "session": "session",
            "prompt": "prompt",
            "gen": "gen",
            "ui": "ui",
            "args": "args",
            "history": "history",
            "last": "last",
            "env": "env",
            "files": "files",
            "model": "model",
            "config": "config",
            "backend": "backend",
            "aliases": "aliases",
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
                summary="Show current runtime/session details",
                usage="/show <topic>",
                handler=UnifiedTuiApp._cmd_show,
                examples=("/show", "/show gen", "/show prompt"),
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
        for alias_name in ("model", "config", "env", "files", "session", "prompt", "gen", "ui", "args", "history", "last", "backend"):
            self.registry.register(
                SlashCommand(
                    name=alias_name,
                    summary=f"Alias for /show {alias_name}",
                    usage=f"/{alias_name}",
                    handler=lambda app, argv, topic=alias_name: app._cmd_show([topic]),
                    include_in_help=False,
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
                aliases=("quit",),
                summary="Exit TUI",
                usage="/exit",
                handler=UnifiedTuiApp._cmd_exit,
            )
        )

    def _cmd_help(self, argv: list[str]) -> str:
        if not argv:
            lines = ["Available commands:"]
            for cmd in self.registry.all_commands():
                if not cmd.include_in_help:
                    continue
                lines.append(f"/{cmd.name}: {cmd.summary}")
            lines.append("")
            lines.append("Try: /help <command>")
            lines.append("Alias map: /show aliases")
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
                "Usage: /show <topic>",
                "Topics: " + ", ".join(sorted(self.show_topics.keys())),
                "Examples:",
                "  /show session",
                "  /show gen",
                "  /show prompt",
            ]
            return "\n".join(lines)
        topic = argv[0].lower()
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
        if len(argv) > 1 and argv[1] in {"?", "--help"}:
            return f"{handler.usage}\n{handler.summary}"
        return handler.handler(self, argv[1:])

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
        chat_template = getattr(args, "chat_template", None) if backend == "hf" else None
        lines = [
            f"backend: {backend}",
            f"user_prefix: {args.user_prefix!r}",
            "prompt_prefix: N/A",
            f"prompt_mode: {prompt_mode if prompt_mode is not None else 'N/A'}",
            f"chat_template: {chat_template if chat_template else 'N/A'}",
        ]
        return "\n".join(lines)

    def _cmd_clear(self, argv: list[str]) -> str:
        del argv
        self.transcript.remove_children()
        self.messages = []
        if self.runtime.args.system:
            self.messages.append({"role": "system", "content": self.runtime.args.system})
        self.pending_assistant = None
        return "Transcript and conversation history cleared."

    def _cmd_exit(self, argv: list[str]) -> str:
        del argv
        self.exit()
        return "Exiting."

    def _show_session(self, argv: list[str]) -> str:
        del argv
        lines = [
            f"backend: {self.runtime.session.backend_name}",
            f"model_id: {self.runtime.session.resolved_model_id}",
            f"config_path: {self.runtime.args._config_path or '(none)'}",
            f"is_generating: {self.is_generating}",
            f"follow_output: {self.follow_output}",
            f"transcript_widgets: {len(self.transcript.children)}",
            f"scroll_y: {float(self.transcript.scroll_y):.1f}/{float(self.transcript.max_scroll_y):.1f}",
        ]
        return "\n".join(lines)

    def _show_prompt(self, argv: list[str]) -> str:
        del argv
        return self._cmd_system([]) + "\n" + self._cmd_prefix([])

    def _show_gen(self, argv: list[str]) -> str:
        del argv
        args = self.runtime.args
        backend = self.runtime.session.backend_name
        cli_overrides = set(getattr(args, "_cli_overrides", set()) or set())
        config_keys = set(getattr(args, "_config_keys", set()) or set())
        user_set = cli_overrides | config_keys

        if backend == "hf":
            keys = [
                "max_new_tokens",
                "temperature",
                "top_p",
                "top_k",
                "stop_strings",
                "repetition_penalty",
                "typical_p",
                "min_p",
                "max_time",
                "num_beams",
                "no_repeat_ngram_size",
            ]
            sent = {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "num_beams": args.num_beams,
                "no_repeat_ngram_size": args.no_repeat_ngram_size,
            }
            if args.top_k is not None:
                sent["top_k"] = args.top_k
            if args.stop_strings is not None:
                sent["stop_strings"] = args.stop_strings
            if args.typical_p is not None:
                sent["typical_p"] = args.typical_p
            if args.min_p is not None:
                sent["min_p"] = args.min_p
            if args.max_time is not None:
                sent["max_time"] = args.max_time
            deferred = []
            for optional_key in ("top_k", "stop_strings", "typical_p", "min_p", "max_time"):
                if optional_key not in sent:
                    deferred.append(optional_key)
        elif backend == "gguf":
            keys = [
                "max_new_tokens",
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "typical_p",
                "repetition_penalty",
                "stop_strings",
                "n_ctx",
                "n_gpu_layers",
                "prompt_mode",
                "chat_template",
            ]
            sent = {
                "max_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
            if args.top_k is not None:
                sent["top_k"] = args.top_k
            if args.min_p is not None:
                sent["min_p"] = args.min_p
            if args.typical_p is not None:
                sent["typical_p"] = args.typical_p
            if args.repetition_penalty not in (None, 1.0):
                sent["repeat_penalty"] = args.repetition_penalty
            if args.stop_strings is not None:
                sent["stop"] = args.stop_strings
            deferred = []
            for optional_key, sent_key in (
                ("top_k", "top_k"),
                ("min_p", "min_p"),
                ("typical_p", "typical_p"),
                ("repetition_penalty", "repeat_penalty"),
                ("stop_strings", "stop"),
            ):
                if sent_key not in sent:
                    deferred.append(optional_key)
        elif backend == "exl2":
            keys = [
                "max_new_tokens",
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "typical_p",
                "repetition_penalty",
                "frequency_penalty",
                "presence_penalty",
                "stop_strings",
                "max_seq_len",
                "min_free_tokens",
                "gpu_split",
                "cache_type",
                "exl2_stop_tokens",
                "exl2_repeat_streak_max",
            ]
            sent = {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": (args.top_k or 0),
                "min_p": (args.min_p or 0),
                "typical": (args.typical_p or 0),
                "repetition_penalty": args.repetition_penalty,
                "frequency_penalty": (args.frequency_penalty or 0.0),
                "presence_penalty": (args.presence_penalty or 0.0),
                "max_seq_len": args.max_seq_len,
                "min_free_tokens": args.min_free_tokens,
                "cache_type": args.cache_type,
                "exl2_repeat_streak_max": args.exl2_repeat_streak_max,
            }
            deferred = []
            if args.stop_strings:
                sent["stop_strings"] = args.stop_strings
            if args.exl2_stop_tokens:
                sent["exl2_stop_tokens"] = args.exl2_stop_tokens
            else:
                deferred.append("exl2_stop_tokens(auto:<end_of_turn>,<start_of_turn>,eos)")
            if args.gpu_split and args.gpu_split != "auto":
                sent["gpu_split"] = args.gpu_split
            else:
                deferred.append("gpu_split(full single-GPU load)")
        else:
            keys = ["max_new_tokens", "temperature", "top_p", "top_k", "stop_strings", "ollama_think"]
            sent = {}
            if "temperature" in user_set and args.temperature is not None:
                sent["options.temperature"] = args.temperature
            if "top_p" in user_set and args.top_p is not None:
                sent["options.top_p"] = args.top_p
            if "top_k" in user_set and args.top_k is not None:
                sent["options.top_k"] = args.top_k
            if "max_new_tokens" in user_set and args.max_new_tokens is not None:
                sent["options.num_predict"] = args.max_new_tokens
            if "stop_strings" in user_set and args.stop_strings is not None:
                sent["options.stop"] = args.stop_strings
            if args.ollama_think in {"true", "false"}:
                sent["think"] = (args.ollama_think == "true")
            deferred = []
            if "options.temperature" not in sent:
                deferred.append("temperature")
            if "options.top_p" not in sent:
                deferred.append("top_p")
            if "options.top_k" not in sent:
                deferred.append("top_k")
            if "options.num_predict" not in sent:
                deferred.append("max_new_tokens")
            if "options.stop" not in sent:
                deferred.append("stop_strings")
            if "think" not in sent:
                deferred.append("ollama_think(auto)")

        args_block = {key: getattr(args, key, None) for key in keys}
        lines = [
            "args.*",
            *[f"  {k}: {args_block[k]!r}" for k in keys],
            "sent.*",
        ]
        if sent:
            lines.extend([f"  {k}: {v!r}" for k, v in sent.items()])
        else:
            lines.append("  (none)")
        lines.append("deferred")
        if deferred:
            lines.extend([f"  {k}" for k in deferred])
        else:
            lines.append("  (none)")
        return "\n".join(lines)

    def _show_ui(self, argv: list[str]) -> str:
        del argv
        args = self.runtime.args
        lines = [
            f"show_thinking: {args.show_thinking}",
            f"no_animate_thinking: {args.no_animate_thinking}",
            f"scroll_lines: {args.scroll_lines}",
            f"ui_tick_ms: {args.ui_tick_ms}",
            f"ui_max_events_per_tick: {args.ui_max_events_per_tick}",
            f"follow_output: {self.follow_output}",
        ]
        return "\n".join(lines)

    def _show_args(self, argv: list[str]) -> str:
        del argv
        data = {}
        for key, value in vars(self.runtime.args).items():
            if key.startswith("_"):
                continue
            data[key] = value
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)

    def _show_history(self, argv: list[str]) -> str:
        del argv
        roles = [msg.get("role", "?") for msg in self.messages[-10:]]
        lines = [
            f"turn_records: {len(self.turn_records)}",
            f"messages: {len(self.messages)}",
            f"last_roles(10): {roles}",
        ]
        return "\n".join(lines)

    def _show_last(self, argv: list[str]) -> str:
        del argv
        if not self.turn_records:
            return "No completed turns yet."
        last = self.turn_records[-1]
        lines = [
            f"backend: {last.backend}",
            f"model_id: {last.model_id}",
            f"ended_in_think: {last.ended_in_think}",
            f"timing.elapsed: {last.timing.get('elapsed')}",
            f"len(raw): {len(last.raw)}",
            f"len(think): {len(last.think)}",
            f"len(answer): {len(last.answer)}",
        ]
        return "\n".join(lines)

    def _show_env(self, argv: list[str]) -> str:
        del argv
        lines = [
            f"OLLAMA_HOST: {os.environ.get('OLLAMA_HOST', '(unset)')}",
            f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}",
        ]
        return "\n".join(lines)

    def _show_files(self, argv: list[str]) -> str:
        del argv
        args = self.runtime.args
        lines = [f"config_path: {args._config_path or '(none)'}"]
        if args.system_file:
            lines.append(
                f"system_file: {resolve_path_maybe_relative(args.system_file, config_path=args._config_path)}"
            )
        else:
            lines.append("system_file: (none)")
        if args.save_transcript:
            lines.append(
                f"save_transcript: {resolve_path_maybe_relative(args.save_transcript, config_path=args._config_path)}"
            )
        else:
            lines.append("save_transcript: (none)")
        return "\n".join(lines)

    def _show_model(self, argv: list[str]) -> str:
        del argv
        return "\n".join(
            [
                f"backend: {self.runtime.session.backend_name}",
                f"model_id: {self.runtime.session.resolved_model_id}",
            ]
        )

    def _show_config(self, argv: list[str]) -> str:
        del argv
        return f"config_path: {self.runtime.args._config_path or '(none)'}"

    def _show_backend(self, argv: list[str]) -> str:
        del argv
        session = self.runtime.session
        lines = [
            f"backend: {session.backend_name}",
            f"resolved_model_id: {session.resolved_model_id}",
        ]
        describe = getattr(session, "describe", None)
        info = getattr(session, "get_info", None)
        extra = None
        if callable(describe):
            extra = describe()
        elif callable(info):
            extra = info()
        if isinstance(extra, dict):
            for key in sorted(extra.keys()):
                lines.append(f"{key}: {extra[key]}")
        return "\n".join(lines)

    def _show_aliases(self, argv: list[str]) -> str:
        del argv
        lines = ["Command aliases:"]
        for name in sorted(self.registry.canonical_names()):
            cmd = self.registry.resolve(name)
            if cmd is None or not cmd.aliases:
                continue
            lines.append(f"  /{name}: " + ", ".join(f"/{alias}" for alias in cmd.aliases))

        lines.append("")
        lines.append("Show-topic aliases:")
        for alias in sorted(self.show_aliases.keys()):
            target = self.show_aliases[alias]
            if alias == target:
                continue
            lines.append(f"  {alias} -> {target}")

        lines.append("")
        lines.append("Command shortcuts to /show <topic>:")
        for name in sorted(self.registry.canonical_names()):
            cmd = self.registry.resolve(name)
            if cmd is None:
                continue
            if cmd.summary.startswith("Alias for /show "):
                lines.append(f"  /{name} -> {cmd.summary.removeprefix('Alias for ')}")
        return "\n".join(lines)

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
