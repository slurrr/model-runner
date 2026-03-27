#!/usr/bin/env python3
"""Build an OmniCoder calibration dataset from SWE-smith trajectories."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a jsonl calibration file from SWE-smith trajectories."
    )
    parser.add_argument(
        "--dataset",
        default="SWE-bench/SWE-smith-trajectories",
        help="Hugging Face dataset id.",
    )
    parser.add_argument(
        "--split",
        default="train[:512]",
        help="Datasets split expression to load.",
    )
    parser.add_argument(
        "--output",
        default="~/data/model-runner/calibration/omnicoder_calibration.jsonl",
        help="Where to write the jsonl calibration file.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=512,
        help="Maximum number of rows to write.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode so rows can be processed without full dataset materialization.",
    )
    return parser.parse_args()


def stringify_message(message: Any) -> str:
    if isinstance(message, str):
        return message.strip()
    if isinstance(message, dict):
        role = message.get("role") or message.get("speaker") or message.get("type") or "message"
        content = message.get("content") or message.get("text") or message.get("value") or message.get("message")
        if isinstance(content, list):
            pieces = []
            for item in content:
                if isinstance(item, dict):
                    piece = item.get("text") or item.get("content") or json.dumps(item, ensure_ascii=False)
                else:
                    piece = str(item)
                if piece:
                    pieces.append(str(piece).strip())
            content_text = "\n".join(piece for piece in pieces if piece)
        elif content is None:
            content_text = json.dumps(message, ensure_ascii=False)
        else:
            content_text = str(content).strip()
        return f"[{role}]\n{content_text}".strip()
    if isinstance(message, list):
        return "\n\n".join(filter(None, (stringify_message(item) for item in message)))
    return str(message).strip()


def extract_text(row: dict[str, Any]) -> str:
    preferred_keys = [
        "messages",
        "trajectory",
        "conversation",
        "transcript",
        "history",
        "prompt",
        "problem_statement",
        "issue",
        "instance_text",
    ]
    for key in preferred_keys:
        if key not in row or row[key] in (None, ""):
            continue
        value = row[key]
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("[") or text.startswith("{"):
                try:
                    parsed = json.loads(text)
                except Exception:
                    parsed = None
                if parsed is not None:
                    text = stringify_message(parsed)
            if text:
                return text
        else:
            text = stringify_message(value)
            if text:
                return text

    # Fallback: stringify the whole row without obvious ids.
    reduced = {
        key: value
        for key, value in row.items()
        if key not in {"id", "instance_id", "problem_id", "patch", "test_patch", "FAIL_TO_PASS"}
    }
    return stringify_message(reduced)


def main() -> int:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("datasets is not installed. Run `.venv/bin/python -m pip install datasets`.")

    ds = load_dataset(args.dataset, split=args.split, streaming=args.streaming)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in ds:
            text = extract_text(row)
            if not text:
                continue
            fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1
            if written >= args.max_rows:
                break

    if written == 0:
        raise SystemExit("No calibration rows were written. Inspect the dataset schema and extraction logic.")

    print(json.dumps({
        "dataset": args.dataset,
        "split": args.split,
        "output": str(out_path.resolve()),
        "rows_written": written,
    }, indent=2))
    sys.stdout.flush()
    if args.streaming:
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
