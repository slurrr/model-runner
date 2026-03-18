from __future__ import annotations

import argparse
import json
import sys

from tui_app.transports.openai_http import compat_chat_completion_request, normalize_openai_base_url, resolve_model_once


def _resolve_model(base_url: str, model: str, timeout_s: float, api_key: str) -> str:
    if model.strip():
        return model.strip()
    return resolve_model_once(base_url, timeout_s=timeout_s, api_key=api_key)


def _tool_schema() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Return the text unchanged.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _build_messages() -> list[dict[str, object]]:
    return [
        {
            "role": "system",
            "content": "You are a tool-calling assistant. When the user asks you to use echo, respond with a tool call only.",
        },
        {
            "role": "user",
            "content": "Call the echo tool with text exactly equal to smoke-test.",
        },
    ]


def _build_payload(model: str, *, stream: bool) -> dict[str, object]:
    return {
        "model": model,
        "stream": stream,
        "messages": _build_messages(),
        "tools": _tool_schema(),
        "tool_choice": {"type": "function", "function": {"name": "echo"}},
        "max_tokens": 256,
        "temperature": 0.0,
    }


def _validate_stream(result: dict[str, object]) -> list[str]:
    issues: list[str] = []
    tool_calls = result.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        issues.append("stream: no tool_calls accumulated from SSE deltas")
    else:
        first = tool_calls[0]
        if str(first.get("name") or "") != "echo":
            issues.append(f"stream: expected tool name echo, got {first.get('name')!r}")
        if "smoke-test" not in str(first.get("arguments") or ""):
            issues.append("stream: tool arguments did not contain smoke-test")
    finish_reason = result.get("finish_reason")
    if finish_reason not in {"tool_calls", "stop"}:
        issues.append(f"stream: unexpected finish_reason {finish_reason!r}")
    return issues


def _validate_nonstream(result: dict[str, object]) -> list[str]:
    issues: list[str] = []
    tool_calls = result.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        issues.append("nonstream: no message.tool_calls in response")
    else:
        first = tool_calls[0]
        if str(first.get("name") or "") != "echo":
            issues.append(f"nonstream: expected tool name echo, got {first.get('name')!r}")
        if "smoke-test" not in str(first.get("arguments") or ""):
            issues.append("nonstream: tool arguments did not contain smoke-test")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenAI-compatible tool-call compatibility smoke check.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    args = parser.parse_args()

    base_url = normalize_openai_base_url(args.base_url)
    model = _resolve_model(base_url, args.model, args.timeout_s, args.api_key)

    stream_payload = _build_payload(model, stream=True)
    nonstream_payload = _build_payload(model, stream=False)

    stream_result = compat_chat_completion_request(
        base_url=base_url,
        payload=stream_payload,
        timeout_s=args.timeout_s,
        api_key=args.api_key,
    )
    nonstream_result = compat_chat_completion_request(
        base_url=base_url,
        payload=nonstream_payload,
        timeout_s=args.timeout_s,
        api_key=args.api_key,
    )

    issues = _validate_stream(stream_result) + _validate_nonstream(nonstream_result)

    report = {
        "base_url": base_url,
        "model": model,
        "stream": {
            "finish_reason": stream_result.get("finish_reason"),
            "usage": stream_result.get("usage"),
            "tool_calls": stream_result.get("tool_calls"),
        },
        "nonstream": {
            "finish_reason": nonstream_result.get("finish_reason"),
            "usage": nonstream_result.get("usage"),
            "tool_calls": nonstream_result.get("tool_calls"),
        },
        "ok": not issues,
        "issues": issues,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
