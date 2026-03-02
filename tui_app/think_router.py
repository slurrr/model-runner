from __future__ import annotations


class ThinkRouter:
    def __init__(self):
        self.start_markers = [
            "<think>",
            "<|begin_of_thought|>",
            "<｜begin_of_thought｜>",
            "<｜begin▁of▁thought｜>",
        ]
        self.end_markers = [
            "</think>",
            "<|end_of_thought|>",
            "<｜end_of_thought｜>",
            "<｜end▁of▁thought｜>",
        ]
        self.mode = "answer"
        self.buffer = ""
        all_markers = self.start_markers + self.end_markers
        self.max_marker_len = max(len(m) for m in all_markers)

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

    def feed(self, piece: str) -> list[tuple[str, str]]:
        self.buffer += piece
        events: list[tuple[str, str]] = []

        while True:
            if self.mode == "answer":
                idx, marker = self._find_first(self.buffer, self.start_markers)
                if idx == -1:
                    keep = self.max_marker_len
                    if len(self.buffer) > keep:
                        emit = self.buffer[:-keep]
                        if emit:
                            events.append(("answer", emit))
                        self.buffer = self.buffer[-keep:]
                    break

                before = self.buffer[:idx]
                if before:
                    events.append(("answer", before))
                self.buffer = self.buffer[idx + len(marker) :]
                self.mode = "think"
            else:
                idx, marker = self._find_first(self.buffer, self.end_markers)
                if idx == -1:
                    keep = self.max_marker_len
                    if len(self.buffer) > keep:
                        emit = self.buffer[:-keep]
                        if emit:
                            events.append(("think", emit))
                        self.buffer = self.buffer[-keep:]
                    break

                before = self.buffer[:idx]
                if before:
                    events.append(("think", before))
                self.buffer = self.buffer[idx + len(marker) :]
                self.mode = "answer"

        return events

    def flush(self) -> list[tuple[str, str]]:
        events: list[tuple[str, str]] = []
        if self.buffer:
            events.append((self.mode, self.buffer))
        self.buffer = ""
        return events
