import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request


class ThinkFilter:
    def __init__(self, strict_prefix_strip: bool = False):
        self.start_end_map = {
            "<think>": "</think>",
            "<|begin_of_thought|>": "<|end_of_thought|>",
            "<｜begin_of_thought｜>": "<｜end_of_thought｜>",
            "<｜begin▁of▁thought｜>": "<｜end▁of▁thought｜>",
        }
        self.start_markers = list(self.start_end_map.keys())
        self.end_markers = list(set(self.start_end_map.values()) | {"</think>"})
        self.max_marker_len = max(len(x) for x in self.start_end_map) + 8
        self.in_think = False
        self.current_end_marker = ""
        self.buffer = ""
        # Some models emit hidden reasoning without an opening marker and only end with </think>.
        # Hold early output briefly to detect and strip that prefix if present.
        self.implicit_prefix_mode = True
        self.implicit_prefix_buffer = ""
        self.implicit_prefix_probe_limit = None if strict_prefix_strip else 8192

    @staticmethod
    def _find_first(text: str, markers: list[str]) -> tuple[int, str]:
        first_idx = -1
        first_marker = ""
        for marker in markers:
            idx = text.find(marker)
            if idx != -1 and (first_idx == -1 or idx < first_idx):
                first_idx = idx
                first_marker = marker
        return first_idx, first_marker

    def feed(self, text: str) -> str:
        if self.implicit_prefix_mode:
            self.implicit_prefix_buffer += text
            end_idx, end_marker = self._find_first(self.implicit_prefix_buffer, self.end_markers)
            if end_idx != -1:
                remainder = self.implicit_prefix_buffer[end_idx + len(end_marker) :]
                self.implicit_prefix_mode = False
                self.implicit_prefix_buffer = ""
                if not remainder:
                    return ""
                return self.feed(remainder)

            if self.implicit_prefix_probe_limit is None:
                return ""

            if len(self.implicit_prefix_buffer) <= self.implicit_prefix_probe_limit:
                return ""

            release = self.implicit_prefix_buffer
            self.implicit_prefix_mode = False
            self.implicit_prefix_buffer = ""
            return self.feed(release)

        self.buffer += text
        out = []

        while True:
            if not self.in_think:
                start_idx, start_marker = self._find_first(self.buffer, self.start_markers)
                if start_idx == -1:
                    keep = self.max_marker_len
                    if len(self.buffer) > keep:
                        out.append(self.buffer[:-keep])
                        self.buffer = self.buffer[-keep:]
                    break

                out.append(self.buffer[:start_idx])
                self.buffer = self.buffer[start_idx + len(start_marker) :]
                self.in_think = True
                self.current_end_marker = self.start_end_map[start_marker]
            else:
                end_idx = self.buffer.find(self.current_end_marker)
                if end_idx == -1:
                    keep = max(self.max_marker_len, len(self.current_end_marker) + 8)
                    if len(self.buffer) > keep:
                        self.buffer = self.buffer[-keep:]
                    break

                self.buffer = self.buffer[end_idx + len(self.current_end_marker) :]
                self.in_think = False
                self.current_end_marker = ""

        return "".join(out)

    def flush(self) -> str:
        if self.implicit_prefix_mode:
            remaining = self.implicit_prefix_buffer
            self.implicit_prefix_mode = False
            self.implicit_prefix_buffer = ""
            return remaining
        if self.in_think:
            self.buffer = ""
            self.current_end_marker = ""
            return ""
        remaining = self.buffer
        self.buffer = ""
        return remaining


def can_reach_ollama_host(host: str, timeout: int) -> bool:
    url = host.rstrip("/") + "/api/tags"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def detect_wsl_gateway_ip() -> str:
    try:
        out = subprocess.check_output(["ip", "route"], text=True, timeout=2)
    except Exception:
        return ""
    for line in out.splitlines():
        if line.startswith("default "):
            parts = line.split()
            if "via" in parts:
                via_idx = parts.index("via")
                if via_idx + 1 < len(parts):
                    return parts[via_idx + 1].strip()
    return ""


def resolve_host(cli_host: str | None, timeout: int) -> str:
    if cli_host:
        return cli_host.rstrip("/")

    candidates = []
    env_host = os.environ.get("OLLAMA_HOST", "").strip()
    if env_host:
        candidates.append(env_host)

    candidates.extend(
        [
            "http://127.0.0.1:11434",
            "http://localhost:11434",
        ]
    )

    gateway_ip = detect_wsl_gateway_ip()
    if gateway_ip:
        candidates.append(f"http://{gateway_ip}:11434")

    seen = set()
    for candidate in candidates:
        c = candidate.rstrip("/")
        if c in seen:
            continue
        seen.add(c)
        if can_reach_ollama_host(c, timeout=min(timeout, 2)):
            return c

    # fallback if probing fails
    return "http://127.0.0.1:11434"


def stream_chat(
    host: str,
    model: str,
    messages: list[dict],
    timeout: int,
    hide_think: bool,
    think: str,
    strict_think_strip: bool,
) -> tuple[str, str]:
    url = host.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    if think in {"true", "false"}:
        payload["think"] = (think == "true")
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    filter_state = ThinkFilter(strict_prefix_strip=strict_think_strip)
    full_text = []
    shown_text = []

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                obj = json.loads(line)
                token = obj.get("message", {}).get("content", "")
                thinking_token = obj.get("message", {}).get("thinking", "")
                if not isinstance(token, str):
                    token = str(token)
                if not isinstance(thinking_token, str):
                    thinking_token = str(thinking_token)
                full_text.append(token)

                if hide_think:
                    filtered = filter_state.feed(token)
                    if filtered:
                        shown_text.append(filtered)
                        print(filtered, end="", flush=True)
                else:
                    if thinking_token:
                        shown_text.append(thinking_token)
                        print(thinking_token, end="", flush=True)
                    if token:
                        shown_text.append(token)
                        print(token, end="", flush=True)

                if obj.get("done"):
                    break
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama API request failed: {exc}") from exc

    if hide_think:
        tail = filter_state.flush()
        if tail:
            shown_text.append(tail)
            print(tail, end="", flush=True)

    print()
    return "".join(full_text), "".join(shown_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight Ollama chat wrapper with optional <think> filtering.")
    parser.add_argument("model", help="Ollama model name, e.g. my-deepseek-r1")
    parser.add_argument(
        "--host",
        default=None,
        help="Ollama host URL. If omitted, auto-detects from OLLAMA_HOST/localhost/WSL gateway.",
    )
    parser.add_argument("--timeout", type=int, default=600, help="Request timeout in seconds")
    parser.add_argument(
        "--think",
        choices=["auto", "true", "false"],
        default="false",
        help="Set Ollama API think mode (default: false).",
    )
    parser.add_argument(
        "--show-think",
        action="store_true",
        help="Show <think> blocks instead of filtering them",
    )
    parser.add_argument(
        "--strict-think-strip",
        action="store_true",
        help="If enabled, hide all initial output until a think end marker is seen.",
    )
    parser.add_argument(
        "--system",
        default="",
        help="Optional system instruction added at session start",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hide_think = not args.show_think
    host = resolve_host(args.host, args.timeout)
    if not can_reach_ollama_host(host, timeout=min(args.timeout, 2)):
        print(f"Could not reach Ollama at {host}")
        print("Set --host explicitly or export OLLAMA_HOST.")
        sys.exit(1)

    messages = []
    if args.system.strip():
        messages.append({"role": "system", "content": args.system.strip()})

    print(f"Model: {args.model}")
    print(f"Ollama host: {host}")
    print(f"think mode: {args.think}")
    print("Commands: /exit, /quit, /clear")

    while True:
        try:
            user_text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_text.lower() in {"exit", "quit", "/exit", "/quit"}:
            break
        if user_text.lower() == "/clear":
            messages = []
            if args.system.strip():
                messages.append({"role": "system", "content": args.system.strip()})
            print("Conversation cleared.")
            continue
        if not user_text:
            continue

        messages.append({"role": "user", "content": user_text})
        try:
            assistant_raw, assistant_shown = stream_chat(
                host=host,
                model=args.model,
                messages=messages,
                timeout=args.timeout,
                hide_think=hide_think,
                think=args.think,
                strict_think_strip=args.strict_think_strip,
            )
        except RuntimeError as exc:
            print(exc)
            messages.pop()
            continue

        # Keep history aligned with what the user actually saw when filtering is enabled.
        assistant_for_history = assistant_shown if hide_think else assistant_raw
        messages.append({"role": "assistant", "content": assistant_for_history})


if __name__ == "__main__":
    main()
