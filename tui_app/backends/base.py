from __future__ import annotations

from typing import Callable, Protocol

from tui_app.events import AnswerDelta, Error, Finish, Meta, ThinkDelta, TurnStart

Event = TurnStart | ThinkDelta | AnswerDelta | Meta | Error | Finish
EventEmitter = Callable[[Event], None]


class BackendSession(Protocol):
    backend_name: str
    resolved_model_id: str

    def generate_turn(self, turn_id: int, messages: list[dict[str, object]], emit: EventEmitter) -> None:
        ...

    def get_recent_logs(self, n: int = 80, sources: list[str] | None = None) -> list[str]:
        ...

    def list_log_sources(self) -> list[str]:
        ...
