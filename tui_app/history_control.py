from __future__ import annotations


def build_assistant_history_content(*, think: str, answer: str, strip_think: bool) -> str | None:
    think_text = think or ""
    answer_text = answer or ""
    think_has_content = bool(think_text.strip())
    answer_has_content = bool(answer_text.strip())
    if strip_think:
        return answer_text if answer_has_content else None
    if think_has_content and answer_has_content:
        return f"<think>{think_text}</think>{answer_text}"
    if think_has_content:
        return f"<think>{think_text}</think>"
    return answer_text if answer_has_content else None


def append_assistant_history(
    messages: list[dict[str, object]],
    *,
    think: str,
    answer: str,
    strip_think: bool,
) -> list[dict[str, object]]:
    out = list(messages)
    content = build_assistant_history_content(think=think, answer=answer, strip_think=strip_think)
    if content is not None and content.strip():
        out.append({"role": "assistant", "content": content})
    return out
