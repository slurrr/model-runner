from __future__ import annotations

import collections
import os
import threading
import time


class FileLogger:
    def __init__(self, path: str | None, label: str, *, max_tail: int = 120):
        raw = (path or "").strip()
        self.path = os.path.abspath(os.path.expanduser(raw)) if raw else ""
        self.label = label
        self._fh = None
        if self.path:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()
        self._tail: collections.deque[str] = collections.deque(maxlen=max(1, int(max_tail)))

    @classmethod
    def from_value(cls, path: str | None, label: str) -> FileLogger:
        return cls(path, label)

    def log(self, message: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{self.label}] {message.rstrip()}\n"
        with self._lock:
            if self._fh is not None:
                self._fh.write(line)
                self._fh.flush()
            self._tail.append(line.rstrip("\n"))

    def get_recent_logs(self, n: int = 80) -> list[str]:
        size = max(1, int(n))
        with self._lock:
            data = list(self._tail)
        if size >= len(data):
            return data
        return data[-size:]

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
