from __future__ import annotations

import collections
import os
import re
import threading
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


_QUERY_SECRET_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "bearer",
    "key",
    "password",
    "sig",
    "signature",
    "token",
}


class FileLogger:
    def __init__(
        self,
        path: str | None,
        default_source: str,
        *,
        config_path: str | None = None,
        max_tail: int = 500,
        max_line_bytes: int = 8 * 1024,
    ):
        raw = (path or "").strip()
        self.path = self._resolve_path(raw, config_path=config_path) if raw else ""
        self.default_source = (default_source or "backend").strip() or "backend"
        self.max_line_bytes = max(256, int(max_line_bytes))
        self._fh = None
        self._sink_failed = False
        self._warned_keys: set[str] = set()
        self._lock = threading.Lock()
        self._tail: collections.deque[tuple[str, str]] = collections.deque(maxlen=max(1, int(max_tail)))
        self._open_sink()

    @classmethod
    def from_value(
        cls,
        path: str | None,
        label: str,
        *,
        config_path: str | None = None,
        max_tail: int = 500,
    ) -> FileLogger:
        return cls(path, label, config_path=config_path, max_tail=max_tail)

    @staticmethod
    def _resolve_path(path: str, *, config_path: str | None = None) -> str:
        expanded = os.path.expanduser(path)
        if os.path.isabs(expanded):
            return expanded
        if config_path:
            cfg_dir = os.path.dirname(config_path)
            if cfg_dir:
                return os.path.abspath(os.path.join(cfg_dir, expanded))
        return os.path.abspath(expanded)

    @staticmethod
    def _ts_utc() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    @staticmethod
    def _normalize_single_line(message: str) -> str:
        return str(message).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n").strip()

    @staticmethod
    def _redact_query_secrets(text: str) -> str:
        def _replace_url(match: re.Match[str]) -> str:
            raw = match.group(0)
            try:
                parts = urlsplit(raw)
            except Exception:
                return raw
            if not parts.query:
                return raw
            changed = False
            query = []
            for key, value in parse_qsl(parts.query, keep_blank_values=True):
                if key.lower() in _QUERY_SECRET_KEYS:
                    query.append((key, "***"))
                    changed = True
                else:
                    query.append((key, value))
            if not changed:
                return raw
            return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))

        return re.sub(r"https?://[^\s]+", _replace_url, text)

    @classmethod
    def _redact(cls, message: str) -> str:
        text = message
        text = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+", "<image:data_url omitted>", text, flags=re.IGNORECASE)
        text = re.sub(r"(?i)(authorization\s*[:=]\s*)(bearer\s+)?([^\s,;]+)", r"\1***", text)
        text = re.sub(
            r"(?i)\b(openai_api_key|vllm_api_key|api[_-]?key|access_token|token)\b(\s*[:=]\s*)(['\"]?)([^'\"\s,]+)(\3)",
            r"\1\2\3***\5",
            text,
        )
        text = re.sub(
            r"(?i)([?&](?:access_token|api[_-]?key|apikey|auth|authorization|bearer|key|password|sig|signature|token)=)([^&#\s]+)",
            r"\1***",
            text,
        )
        text = cls._redact_query_secrets(text)
        return text

    def _truncate(self, message: str) -> str:
        data = message.encode("utf-8", errors="ignore")
        if len(data) <= self.max_line_bytes:
            return message
        marker = "...<truncated>"
        keep = max(64, self.max_line_bytes - len(marker.encode("utf-8")))
        trimmed = data[:keep].decode("utf-8", errors="ignore")
        return trimmed + marker

    def _render_line(self, source: str, message: str) -> str:
        normalized = self._normalize_single_line(message)
        redacted = self._redact(normalized)
        truncated = self._truncate(redacted)
        return f"{self._ts_utc()} [{source}] {truncated}"

    def _open_sink(self) -> None:
        if not self.path:
            return
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")
        except Exception as exc:
            self._fh = None
            self._sink_failed = True
            self.warn_once("sink_open_failed", f"Log file sink disabled: {exc}")

    def log(self, message: str, *, source: str | None = None) -> None:
        src = (source or self.default_source or "backend").strip() or "backend"
        line = self._render_line(src, message)
        warn_message = ""
        with self._lock:
            self._tail.append((src, line))
            if self._fh is not None:
                try:
                    self._fh.write(line + "\n")
                    self._fh.flush()
                except Exception as exc:
                    try:
                        self._fh.close()
                    except Exception:
                        pass
                    self._fh = None
                    self._sink_failed = True
                    warn_message = f"Log file sink disabled after write failure: {exc}"
        if warn_message:
            self.warn_once("sink_write_failed", warn_message)

    def warn_once(self, key: str, message: str, *, source: str = "app") -> None:
        with self._lock:
            if key in self._warned_keys:
                return
            self._warned_keys.add(key)
        self.log(message, source=source)

    def get_recent_logs(self, n: int = 80, sources: list[str] | None = None) -> list[str]:
        size = max(1, int(n))
        allowed = {s.strip() for s in (sources or []) if str(s).strip()}
        with self._lock:
            data = list(self._tail)
        if allowed:
            rows = [line for src, line in data if src in allowed]
        else:
            rows = [line for _, line in data]
        if size >= len(rows):
            return rows
        return rows[-size:]

    def list_log_sources(self) -> list[str]:
        with self._lock:
            sources = {src for src, _ in self._tail}
        return sorted(sources)

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
