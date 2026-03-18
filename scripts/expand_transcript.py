#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


def _usage() -> int:
    print("Usage: expand <path-to-jsonl>", file=sys.stderr)
    return 2


def _read_records(path: Path) -> list[dict]:
    records: list[dict] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{path}:{idx}: invalid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise RuntimeError(f"{path}:{idx}: expected object per line")
        records.append(obj)
    return records


def _output_path(path: Path) -> Path:
    if path.suffix.lower() == ".jsonl":
        return path.with_suffix(".txt")
    return path.with_name(path.name + ".txt")


def _fmt_value(value: object) -> str:
    if value is None:
        return "unavailable"
    return str(value)


def _append_section(lines: list[str], title: str, body: str) -> None:
    if not body.strip():
        return
    lines.append(f"{title}:")
    lines.append(body.rstrip())
    lines.append("")


def _render_record(index: int, record: dict) -> str:
    timing = record.get("timing") if isinstance(record.get("timing"), dict) else {}
    token_counts = record.get("token_counts") if isinstance(record.get("token_counts"), dict) else {}
    throughput = record.get("throughput") if isinstance(record.get("throughput"), dict) else {}

    lines = [
        f"Turn {index}",
        f"backend: {_fmt_value(record.get('backend'))}",
        f"model_id: {_fmt_value(record.get('model_id'))}",
        f"ended_in_think: {_fmt_value(record.get('ended_in_think'))}",
        f"elapsed_s: {_fmt_value(timing.get('elapsed'))}",
        f"prompt_tokens: {_fmt_value(token_counts.get('prompt_tokens'))}",
        f"completion_tokens: {_fmt_value(token_counts.get('completion_tokens'))}",
        f"total_tokens: {_fmt_value(token_counts.get('total_tokens'))}",
        f"tokens_per_s: {_fmt_value(throughput.get('tokens_per_s'))}",
        "",
    ]

    answer = str(record.get("answer") or "")
    think = str(record.get("think") or "")
    raw = str(record.get("raw") or "")

    _append_section(lines, "answer", answer)
    _append_section(lines, "think", think)
    if raw and raw != answer and raw != think:
        _append_section(lines, "raw", raw)

    lines.append("=" * 80)
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        return _usage()

    src = Path(argv[1]).expanduser()
    if not src.is_absolute():
        src = (Path.cwd() / src).resolve()
    if not src.is_file():
        print(f"Input not found: {src}", file=sys.stderr)
        return 1

    try:
        records = _read_records(src)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    dst = _output_path(src)
    body = "\n".join(_render_record(idx, rec) for idx, rec in enumerate(records, start=1))
    if body and not body.endswith("\n"):
        body += "\n"
    dst.write_text(body, encoding="utf-8")
    print(dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
