from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnRecord:
    raw: str
    think: str
    answer: str
    ended_in_think: bool
    backend: str
    model_id: str
    gen: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    token_counts: dict[str, int] | None = None
    throughput: dict[str, float] | None = None
    knobs: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    tool_activity: list[dict[str, Any]] | None = None
    trimmed_messages: list[dict[str, object]] | None = None


@dataclass
class TurnStart:
    turn_id: int


@dataclass
class ThinkDelta:
    turn_id: int
    text: str


@dataclass
class AnswerDelta:
    turn_id: int
    text: str


@dataclass
class Meta:
    turn_id: int
    key: str
    value: Any


@dataclass
class Error:
    turn_id: int
    message: str


@dataclass
class Finish:
    turn_id: int
    record: TurnRecord
