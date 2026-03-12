from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


MeasureFn = Callable[[list[dict[str, object]]], tuple[int, Any]]


@dataclass
class ContextReport:
    strategy: str
    original_messages: int
    kept_messages: int
    dropped_messages: int
    dropped_roles: list[str] = field(default_factory=list)
    fit: bool = True
    context_window: int | None = None
    reserved_generation_tokens: int | None = None
    prompt_budget_tokens: int | None = None
    prompt_tokens: int | None = None
    overflow_retries: int = 0
    system_message_present: bool = False
    system_message_preserved: bool = True
    system_drop_required: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "original_messages": self.original_messages,
            "kept_messages": self.kept_messages,
            "dropped_messages": self.dropped_messages,
            "dropped_roles": list(self.dropped_roles),
            "fit": self.fit,
            "context_window": self.context_window,
            "reserved_generation_tokens": self.reserved_generation_tokens,
            "prompt_budget_tokens": self.prompt_budget_tokens,
            "prompt_tokens": self.prompt_tokens,
            "overflow_retries": self.overflow_retries,
            "system_message_present": self.system_message_present,
            "system_message_preserved": self.system_message_preserved,
            "system_drop_required": self.system_drop_required,
        }


def reserve_generation_tokens(context_window: int | None, requested_max_new_tokens: int | None) -> int | None:
    if context_window is None or context_window <= 1:
        return None
    if requested_max_new_tokens is None or int(requested_max_new_tokens) <= 0:
        return 0
    return max(1, min(int(requested_max_new_tokens), int(context_window) - 1))


def _protected_suffix_start(messages: list[dict[str, object]]) -> int:
    last_user_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "user":
            last_user_idx = idx
            break
    if last_user_idx != -1:
        return last_user_idx
    return max(0, len(messages) - 1)


def _tool_call_id_set(message: dict[str, object]) -> set[str]:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return set()
    ids: set[str] = set()
    for item in tool_calls:
        if not isinstance(item, dict):
            continue
        call_id = item.get("id")
        if isinstance(call_id, str) and call_id.strip():
            ids.add(call_id.strip())
    return ids


def _drop_tool_exchange(messages: list[dict[str, object]], idx: int) -> list[str]:
    assistant = messages[idx]
    call_ids = _tool_call_id_set(assistant)
    if not call_ids:
        role = str(assistant.get("role", "") or "") or "assistant"
        messages.pop(idx)
        return [role]

    dropped_roles = [str(assistant.get("role", "") or "") or "assistant"]
    end = idx + 1
    while end < len(messages):
        message = messages[end]
        role = str(message.get("role", "") or "")
        if role != "tool":
            break
        tool_call_id = str(message.get("tool_call_id", "") or "").strip()
        if tool_call_id not in call_ids:
            break
        dropped_roles.append("tool")
        end += 1
    del messages[idx:end]
    return dropped_roles


def _turn_bundle_end(messages: list[dict[str, object]], start: int, protected_suffix_start: int) -> int:
    end = start + 1
    while end < protected_suffix_start:
        role = str(messages[end].get("role", "") or "")
        if role == "user":
            break
        if role == "assistant" and _tool_call_id_set(messages[end]):
            call_ids = _tool_call_id_set(messages[end])
            end += 1
            while end < protected_suffix_start:
                next_role = str(messages[end].get("role", "") or "")
                if next_role != "tool":
                    break
                tool_call_id = str(messages[end].get("tool_call_id", "") or "").strip()
                if tool_call_id not in call_ids:
                    break
                end += 1
            continue
        end += 1
    return end


def drop_oldest_history_message(messages: list[dict[str, object]]) -> list[str] | None:
    if not messages:
        return None
    protected_suffix_start = _protected_suffix_start(messages)
    for idx, item in enumerate(messages):
        if idx >= protected_suffix_start:
            break
        role = str(item.get("role", "") or "")
        if role == "system":
            continue
        if role == "user":
            end = _turn_bundle_end(messages, idx, protected_suffix_start)
            dropped_roles = [str(msg.get("role", "") or "") or "message" for msg in messages[idx:end]]
            del messages[idx:end]
            return dropped_roles
        if role == "assistant" and _tool_call_id_set(item):
            return _drop_tool_exchange(messages, idx)
        if role in {"user", "assistant", "tool"}:
            messages.pop(idx)
            return [role]
    return None


def trim_messages_to_budget(
    messages: list[dict[str, object]],
    *,
    measure_fn: MeasureFn,
    context_window: int,
    reserved_generation_tokens: int | None,
    strategy: str,
) -> tuple[list[dict[str, object]], int, Any, ContextReport]:
    working = list(messages)
    budget = max(1, int(context_window) - int(reserved_generation_tokens or 0))
    dropped_roles: list[str] = []
    last_count = 0
    last_value: Any = None
    system_present = any(str(msg.get("role", "") or "") == "system" for msg in messages)

    while True:
        last_count, last_value = measure_fn(working)
        if last_count <= budget:
            break
        dropped_roles_chunk = drop_oldest_history_message(working)
        if dropped_roles_chunk is None:
            report = ContextReport(
                strategy=strategy,
                original_messages=len(messages),
                kept_messages=len(working),
                dropped_messages=len(dropped_roles),
                dropped_roles=dropped_roles,
                fit=False,
                context_window=int(context_window),
                reserved_generation_tokens=reserved_generation_tokens,
                prompt_budget_tokens=budget,
                prompt_tokens=last_count,
                system_message_present=system_present,
                system_message_preserved=system_present,
                system_drop_required=system_present,
            )
            return working, last_count, last_value, report
        dropped_roles.extend(dropped_roles_chunk)

    report = ContextReport(
        strategy=strategy,
        original_messages=len(messages),
        kept_messages=len(working),
        dropped_messages=len(dropped_roles),
        dropped_roles=dropped_roles,
        fit=True,
        context_window=int(context_window),
        reserved_generation_tokens=reserved_generation_tokens,
        prompt_budget_tokens=budget,
        prompt_tokens=last_count,
        system_message_present=system_present,
        system_message_preserved=system_present,
        system_drop_required=False,
    )
    return working, last_count, last_value, report


def build_retry_report(
    original_messages: list[dict[str, object]],
    kept_messages: list[dict[str, object]],
    *,
    strategy: str,
    context_window: int | None,
    reserved_generation_tokens: int | None,
    overflow_retries: int,
    fit: bool,
    prompt_tokens: int | None = None,
    dropped_roles: list[str] | None = None,
) -> ContextReport:
    roles = list(dropped_roles or [])
    system_present = any(str(msg.get("role", "") or "") == "system" for msg in original_messages)
    prompt_budget = None
    if context_window is not None:
        prompt_budget = max(1, int(context_window) - int(reserved_generation_tokens or 0))
    return ContextReport(
        strategy=strategy,
        original_messages=len(original_messages),
        kept_messages=len(kept_messages),
        dropped_messages=len(roles),
        dropped_roles=roles,
        fit=fit,
        context_window=context_window,
        reserved_generation_tokens=reserved_generation_tokens,
        prompt_budget_tokens=prompt_budget,
        prompt_tokens=prompt_tokens,
        overflow_retries=int(overflow_retries),
        system_message_present=system_present,
        system_message_preserved=system_present,
        system_drop_required=bool(system_present and not fit),
    )


def build_context_limit_error(report: ContextReport) -> str:
    parts = ["Input is too large even after history trimming."]
    if report.context_window is not None:
        parts.append(f"context_window={report.context_window}")
    if report.prompt_budget_tokens is not None:
        parts.append(f"prompt_budget_tokens={report.prompt_budget_tokens}")
    if report.prompt_tokens is not None:
        parts.append(f"prompt_tokens={report.prompt_tokens}")
    parts.append(f"dropped_history_messages={report.dropped_messages}")
    if report.system_message_present:
        parts.append(f"system_message_preserved={report.system_message_preserved}")
    parts.append("Reduce the current prompt/system prompt or lower max_new_tokens.")
    return " ".join(parts)


def is_context_overflow_text(text: str) -> bool:
    lowered = (text or "").lower()
    patterns = [
        "context window",
        "context length",
        "maximum context length",
        "requested tokens",
        "requested token",
        "too many tokens",
        "max_model_len",
        "input too long",
        "prompt is too long",
        "exceeds the model's context length",
        "exceed the model's context length",
        "exceeds context window",
        "reduce the length",
    ]
    if any(pattern in lowered for pattern in patterns):
        return True
    return "exceed" in lowered and "token" in lowered
