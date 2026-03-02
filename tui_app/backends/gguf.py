from __future__ import annotations

import argparse
import os
import re
import time

from tui_app.backends.base import EventEmitter
from tui_app.events import AnswerDelta, Error, Finish, ThinkDelta, TurnRecord, TurnStart
from tui_app.think_router import ThinkRouter


def normalize_model_path(raw_path: str) -> str:
    path = os.path.expanduser(raw_path.strip())
    win_drive = re.match(r"^([A-Za-z]):[\\/](.*)$", path)
    if win_drive:
        drive = win_drive.group(1).lower()
        rest = win_drive.group(2).replace("\\", "/")
        path = f"/mnt/{drive}/{rest}"
    return os.path.abspath(path)


class GGUFSession:
    backend_name = "gguf"

    def __init__(self, llm, args: argparse.Namespace, resolved_model_id: str):
        self.llm = llm
        self.args = args
        self.resolved_model_id = resolved_model_id

    def _fallback_plain_prompt(self, messages: list[dict[str, str]]) -> str:
        system = self.args.system or "You are a helpful assistant."
        parts = [f"System: {system}"]
        for msg in messages:
            role = msg.get("role", "message").capitalize()
            parts.append(f"{role}: {msg.get('content', '')}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def generate_turn(self, turn_id: int, messages: list[dict[str, str]], emit: EventEmitter) -> None:
        emit(TurnStart(turn_id=turn_id))
        started = time.time()
        router = ThinkRouter(assume_think=self.args.assume_think)

        raw_parts: list[str] = []
        think_parts: list[str] = []
        answer_parts: list[str] = []

        max_tokens = self.args.max_new_tokens
        stop = self.args.stop_strings or None

        try:
            stream = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                stop=stop,
                stream=True,
            )

            for chunk in stream:
                delta = ""
                choices = chunk.get("choices", [])
                if choices:
                    delta_obj = choices[0].get("delta", {}) or {}
                    delta = delta_obj.get("content", "") or ""
                    if not delta:
                        msg = choices[0].get("message", {}) or {}
                        delta = msg.get("content", "") or ""
                if not delta:
                    continue

                raw_parts.append(delta)
                for channel, text in router.feed(delta):
                    if channel == "think":
                        think_parts.append(text)
                        emit(ThinkDelta(turn_id=turn_id, text=text))
                    else:
                        answer_parts.append(text)
                        emit(AnswerDelta(turn_id=turn_id, text=text))
        except Exception as exc:
            try:
                prompt = self._fallback_plain_prompt(messages)
                response = self.llm.create_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    stop=stop or ["\nUser:", "\nSystem:"],
                )
                text = response["choices"][0].get("text", "")
                raw_parts.append(text)
                for channel, part in router.feed(text):
                    if channel == "think":
                        think_parts.append(part)
                        emit(ThinkDelta(turn_id=turn_id, text=part))
                    else:
                        answer_parts.append(part)
                        emit(AnswerDelta(turn_id=turn_id, text=part))
            except Exception as fallback_exc:
                emit(Error(turn_id=turn_id, message=f"Generation failed: {exc}; fallback failed: {fallback_exc}"))
                return

        for channel, text in router.flush():
            if channel == "think":
                think_parts.append(text)
                emit(ThinkDelta(turn_id=turn_id, text=text))
            else:
                answer_parts.append(text)
                emit(AnswerDelta(turn_id=turn_id, text=text))

        ended = time.time()
        record = TurnRecord(
            raw="".join(raw_parts),
            think="".join(think_parts),
            answer="".join(answer_parts).strip(),
            ended_in_think=(router.mode == "think"),
            backend=self.backend_name,
            model_id=self.resolved_model_id,
            gen={
                "max_new_tokens": max_tokens,
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
            },
            timing={"start": started, "end": ended, "elapsed": max(0.0, ended - started)},
            trimmed_messages=messages,
        )
        emit(Finish(turn_id=turn_id, record=record))


def create_session(args: argparse.Namespace) -> GGUFSession:
    model_path = normalize_model_path(args.model_id)
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {model_path}")
    if not model_path.lower().endswith(".gguf"):
        raise RuntimeError(f"Expected a .gguf file, got: {model_path}")

    try:
        from llama_cpp import Llama, _internals

        original_close = _internals.LlamaModel.close

        def safe_close(self):
            if not hasattr(self, "sampler"):
                self.sampler = None
            if not hasattr(self, "custom_samplers"):
                self.custom_samplers = []
            if not hasattr(self, "_exit_stack"):
                return
            try:
                original_close(self)
            except Exception:
                return

        _internals.LlamaModel.close = safe_close
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: llama-cpp-python. Install with: .venv/bin/pip install llama-cpp-python"
        ) from exc

    print(f"Loading GGUF model: {model_path}")
    llm = Llama(
        model_path=model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=False,
    )
    return GGUFSession(llm=llm, args=args, resolved_model_id=model_path)
