from __future__ import annotations

import argparse
import os
import re
import signal
import sys
import threading
import time

from config_utils import apply_machine_model_root, load_config_layers, load_default_config_layers_for_model
from tui_app.app import TuiRuntime, UnifiedTuiApp
from tui_app.backends.exl2 import create_session as create_exl2_session
from tui_app.backends.gguf import create_session as create_gguf_session
from tui_app.backends.hf import create_session as create_hf_session
from tui_app.backends.ollama import create_session as create_ollama_session
from tui_app.backends.openai import create_session as create_openai_session
from tui_app.backends.vllm import create_session as create_vllm_session, shutdown_managed_server
from tui_app.telemetry import TelemetryContext, attach_log_subscribers


def normalize_windows_path(raw: str) -> str:
    path = os.path.expanduser(raw.strip())
    win_drive = re.match(r"^([A-Za-z]):[\\/](.*)$", path)
    if win_drive:
        drive = win_drive.group(1).lower()
        rest = win_drive.group(2).replace("\\", "/")
        path = f"/mnt/{drive}/{rest}"
    return path


def detect_backend(model: str | None, backend_override: str | None) -> str:
    if backend_override:
        return backend_override
    if not model:
        return "hf"

    normalized = normalize_windows_path(model)
    if normalized.startswith("ollama:"):
        return "ollama"
    if normalized.startswith("openai:"):
        return "openai"
    if normalized.startswith("vllm:"):
        return "vllm"
    if normalized.startswith("exl2:"):
        return "exl2"
    if normalized.lower().endswith(".gguf"):
        return "gguf"
    if os.path.isdir(normalized):
        return "hf"
    return "hf"


def sanitize_ollama_stem(name: str) -> str:
    stem = name
    if stem.startswith("ollama:"):
        stem = stem.split(":", 1)[1]
    stem = stem.replace("/", "__").replace(":", "__")
    return stem


def _resolve_path_maybe_relative(path: str, config_path: str | None = None) -> str:
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    if config_path:
        cfg_dir = os.path.dirname(config_path)
        candidate = os.path.abspath(os.path.join(cfg_dir, expanded))
        if os.path.exists(candidate):
            return candidate
    return os.path.abspath(expanded)


def _is_bare_model_stem(value: str | None) -> bool:
    text = (value or "").strip()
    if not text:
        return False
    if text.startswith((".", "~")):
        return False
    if "/" in text or "\\" in text:
        return False
    if ":" in text:
        return False
    return True


def _resolve_gguf_model_id(
    model_id: str | None,
    *,
    model_path: str | None,
    config_path: str | None,
) -> str | None:
    raw_model_path = (model_path or "").strip()
    if not raw_model_path:
        return model_id

    value = (model_id or "").strip()
    if not value:
        return _resolve_path_maybe_relative(raw_model_path, config_path=config_path)
    if value.lower().endswith(".gguf"):
        return model_id
    if _is_bare_model_stem(value):
        return _resolve_path_maybe_relative(raw_model_path, config_path=config_path)
    return model_id


def _load_default_backend_config(model: str, backend: str, profile: str = ""):
    if backend != "ollama":
        data, meta = load_default_config_layers_for_model(
            model,
            backend=backend,
            profile=profile,
            include_machine=True,
        )
        if meta:
            return data, meta.get("base"), meta
        return None, None, None

    candidates = [model]
    if model.startswith("ollama:"):
        candidates.append(model.split(":", 1)[1])
    candidates.append(sanitize_ollama_stem(model))

    for candidate in candidates:
        data, meta = load_default_config_layers_for_model(
            candidate,
            backend=backend,
            profile=profile,
            include_machine=True,
        )
        if meta:
            return data, meta.get("base"), meta
    return None, None, None


def _infer_backend_from_default_config(model: str) -> str | None:
    matches: list[str] = []
    for backend in ("hf", "gguf", "ollama", "exl2", "openai", "vllm"):
        _, path, _ = _load_default_backend_config(model, backend=backend)
        if path:
            matches.append(backend)
    if len(matches) == 1:
        return matches[0]
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified Textual TUI for HF + GGUF + Ollama + EXL2 + OpenAI-compatible + managed vLLM backends"
    )
    parser.add_argument("model_id", nargs="?", help="Model id/path, .gguf path, or ollama:<name>")
    parser.add_argument(
        "backend_hint",
        nargs="?",
        choices=["hf", "gguf", "ollama", "exl2", "openai", "vllm"],
        help="Optional backend shorthand (e.g. `tui Qwen3.5-9B hf`).",
    )
    parser.add_argument("--backend", choices=["hf", "gguf", "ollama", "exl2", "openai", "vllm"], default=None)
    parser.add_argument("--config", default="", help="Config path or name.")
    parser.add_argument("--profile", default="", help="Optional profile name (loads config/profiles/<name>.toml).")
    parser.add_argument("--backend-only", action="store_true", help="Start backend only without the TUI (vLLM managed only).")
    parser.add_argument(
        "--detach-backend",
        action="store_true",
        help="Keep managed backend running after this process exits.",
    )
    parser.add_argument(
        "--shutdown-backend",
        action="store_true",
        help="Stop the detached managed backend referenced by the current vLLM config.",
    )

    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stream", dest="stream", action="store_true")
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--stop-strings", nargs="+", default=None)
    parser.add_argument("--stop-token-ids", nargs="+", type=int, default=None)
    ignore_eos_group = parser.add_mutually_exclusive_group()
    ignore_eos_group.add_argument("--ignore-eos", dest="ignore_eos", action="store_true")
    ignore_eos_group.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
    parser.add_argument("--min-tokens", type=int, default=None)
    parser.add_argument("--best-of", type=int, default=None)
    beam_search_group = parser.add_mutually_exclusive_group()
    beam_search_group.add_argument("--use-beam-search", dest="use_beam_search", action="store_true")
    beam_search_group.add_argument("--no-use-beam-search", dest="use_beam_search", action="store_false")
    parser.add_argument("--length-penalty", type=float, default=None)
    include_stop_group = parser.add_mutually_exclusive_group()
    include_stop_group.add_argument(
        "--include-stop-str-in-output", dest="include_stop_str_in_output", action="store_true"
    )
    include_stop_group.add_argument(
        "--no-include-stop-str-in-output", dest="include_stop_str_in_output", action="store_false"
    )
    skip_special_group = parser.add_mutually_exclusive_group()
    skip_special_group.add_argument("--skip-special-tokens", dest="skip_special_tokens", action="store_true")
    skip_special_group.add_argument("--no-skip-special-tokens", dest="skip_special_tokens", action="store_false")
    spaces_special_group = parser.add_mutually_exclusive_group()
    spaces_special_group.add_argument(
        "--spaces-between-special-tokens", dest="spaces_between_special_tokens", action="store_true"
    )
    spaces_special_group.add_argument(
        "--no-spaces-between-special-tokens", dest="spaces_between_special_tokens", action="store_false"
    )
    parser.add_argument("--truncate-prompt-tokens", type=int, default=None)
    parser.add_argument("--allowed-token-ids", nargs="+", type=int, default=None)
    parser.add_argument("--prompt-logprobs", type=int, default=None)

    parser.add_argument("--system", default="")
    parser.add_argument("--system-file", default="")
    parser.add_argument("--user-prefix", default="")

    parser.add_argument("--show-thinking", action="store_true")
    tool_blocks_group = parser.add_mutually_exclusive_group()
    tool_blocks_group.add_argument("--show-tool-activity", dest="show_tool_activity", action="store_true")
    tool_blocks_group.add_argument("--no-show-tool-activity", dest="show_tool_activity", action="store_false")
    tool_args_group = parser.add_mutually_exclusive_group()
    tool_args_group.add_argument("--show-tool-arguments", dest="show_tool_arguments", action="store_true")
    tool_args_group.add_argument("--no-show-tool-arguments", dest="show_tool_arguments", action="store_false")
    parser.add_argument("--no-animate-thinking", action="store_true")
    parser.add_argument("--save-transcript", default="")
    tools_enabled_group = parser.add_mutually_exclusive_group()
    tools_enabled_group.add_argument("--tools-enabled", dest="tools_enabled", action="store_true")
    tools_enabled_group.add_argument("--no-tools-enabled", dest="tools_enabled", action="store_false")
    parser.add_argument("--tools-mode", choices=["off", "dry_run", "execute"], default="dry_run")
    parser.add_argument("--tools-schema-file", default="")
    parser.add_argument("--tools-tool-choice", default="")
    parser.add_argument("--tools-allow", nargs="+", default=None)
    parser.add_argument("--tools-deny", nargs="+", default=None)
    parser.add_argument("--tools-max-calls-per-turn", type=int, default=3)
    parser.add_argument("--tools-timeout-s", type=float, default=10.0)
    parser.add_argument("--tools-max-result-chars", type=int, default=8000)
    think_mode_group = parser.add_mutually_exclusive_group()
    think_mode_group.add_argument("--assume-think", dest="assume_think", action="store_true")
    think_mode_group.add_argument("--no-assume-think", dest="assume_think", action="store_false")

    parser.add_argument("--scroll-lines", type=int, default=1)
    parser.add_argument("--ui-tick-ms", type=int, default=33)
    parser.add_argument("--ui-max-events-per-tick", type=int, default=120)
    parser.add_argument("--capture-last-request", action="store_true")
    parser.add_argument("--telemetry-jsonl", default="")
    parser.add_argument("--telemetry-sample-interval-s", type=float, default=1.0)

    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--hf-attn", dest="hf_attn_implementation", choices=["eager", "sdpa", "flash_attention_2"], default=None)
    parser.add_argument("--hf-log-file", default="")
    parser.add_argument("--hf-device-map", default="")
    parser.add_argument("--hf-max-memory", default="")
    hf_text_only_group = parser.add_mutually_exclusive_group()
    hf_text_only_group.add_argument("--hf-text-only", dest="hf_text_only", action="store_true")
    hf_text_only_group.add_argument("--no-hf-text-only", dest="hf_text_only", action="store_false")
    hf_low_cpu_mem_group = parser.add_mutually_exclusive_group()
    hf_low_cpu_mem_group.add_argument("--hf-low-cpu-mem-usage", dest="hf_low_cpu_mem_usage", action="store_true")
    hf_low_cpu_mem_group.add_argument(
        "--no-hf-low-cpu-mem-usage", dest="hf_low_cpu_mem_usage", action="store_false"
    )
    parser.add_argument("--prompt-mode", choices=["chat", "plain"], default="chat")
    parser.add_argument("--chat-template", default="")
    history_strip_group = parser.add_mutually_exclusive_group()
    history_strip_group.add_argument("--history-strip-think", dest="history_strip_think", action="store_true")
    history_strip_group.add_argument("--no-history-strip-think", dest="history_strip_think", action="store_false")
    parser.set_defaults(history_strip_think=None)
    parser.add_argument("--max-context-tokens", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--frequency-penalty", type=float, default=None)
    parser.add_argument("--typical-p", type=float, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("-8bit", "--8bit", dest="use_8bit", action="store_true")
    parser.add_argument("-4bit", "--4bit", dest="use_4bit", action="store_true")

    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--gguf-log-file", default="")

    # EXL2 / ExLlamaV2 knobs (TUI only)
    parser.add_argument("--max-seq-len", dest="max_seq_len", type=int, default=None)
    parser.add_argument("--min-free-tokens", type=int, default=256)
    parser.add_argument("--gpu-split", default="")
    parser.add_argument("--cache-type", choices=["fp16", "8bit", "q4", "q6", "q8"], default="fp16")
    parser.add_argument("--exl2-stop-tokens", nargs="+", default=None)
    parser.add_argument("--exl2-repeat-streak-max", type=int, default=64)
    parser.add_argument("--rope-scale", type=float, default=None)
    parser.add_argument("--rope-alpha", type=float, default=None)
    parser.add_argument("--rope-yarn", type=float, default=None)
    parser.add_argument("--low-mem", action="store_true")
    parser.add_argument("--exl2-repo-path", default="")
    parser.add_argument("--exl2-log-file", default="")

    flash_group = parser.add_mutually_exclusive_group()
    flash_group.add_argument("--flash-attn", dest="flash_attn", action="store_true")
    flash_group.add_argument("--no-flash-attn", dest="flash_attn", action="store_false")

    xformers_group = parser.add_mutually_exclusive_group()
    xformers_group.add_argument("--xformers", dest="xformers", action="store_true")
    xformers_group.add_argument("--no-xformers", dest="xformers", action="store_false")

    sdpa_group = parser.add_mutually_exclusive_group()
    sdpa_group.add_argument("--sdpa", dest="sdpa", action="store_true")
    sdpa_group.add_argument("--no-sdpa", dest="sdpa", action="store_false")

    graphs_group = parser.add_mutually_exclusive_group()
    graphs_group.add_argument("--graphs", dest="graphs", action="store_true")
    graphs_group.add_argument("--no-graphs", dest="graphs", action="store_false")

    parser.add_argument("--ollama-host", default=None)
    parser.add_argument("--ollama-timeout", type=int, default=600)
    parser.add_argument("--ollama-think", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--ollama-log-file", default="")
    parser.add_argument("--openai-base-url", default="")
    parser.add_argument("--openai-api-key", default="")
    parser.add_argument("--openai-timeout-s", type=int, default=600)
    parser.add_argument("--openai-log-file", default="")
    parser.add_argument("--vllm-mode", choices=["managed"], default="managed")
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-cmd", default="vllm")
    parser.add_argument("--vllm-extra-args", nargs="*", default=None)
    parser.add_argument("--vllm-served-model-name", default="")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-max-model-len", type=int, default=0)
    parser.add_argument("--vllm-base-url", default="")
    parser.add_argument("--vllm-api-key", default="")
    parser.add_argument("--vllm-timeout-s", type=int, default=600)
    parser.add_argument("--vllm-generation-config", default="auto")
    parser.add_argument("--vllm-attention-backend", default="")
    parser.add_argument("--vllm-dtype", default="")
    vllm_tool_choice_group = parser.add_mutually_exclusive_group()
    vllm_tool_choice_group.add_argument(
        "--vllm-enable-auto-tool-choice", dest="vllm_enable_auto_tool_choice", action="store_true"
    )
    vllm_tool_choice_group.add_argument(
        "--no-vllm-enable-auto-tool-choice", dest="vllm_enable_auto_tool_choice", action="store_false"
    )
    parser.add_argument("--vllm-tool-call-parser", default="")
    parser.add_argument("--vllm-log-file", default="")

    parser.set_defaults(
        stream=True,
        assume_think=None,
        tools_enabled=False,
        show_tool_activity=False,
        show_tool_arguments=False,
    )
    parser.set_defaults(
        flash_attn=None,
        xformers=None,
        sdpa=None,
        graphs=None,
        ignore_eos=None,
        use_beam_search=None,
        include_stop_str_in_output=None,
        skip_special_tokens=None,
        spaces_between_special_tokens=None,
        vllm_enable_auto_tool_choice=None,
        hf_text_only=None,
        hf_low_cpu_mem_usage=None,
    )
    return parser


def _collect_config_defaults(config_data: dict | None) -> dict:
    if not isinstance(config_data, dict):
        return {}
    normalized = dict(config_data)
    # Backward compatibility for early EXL2 configs.
    if "typical_p" not in normalized and "typical" in normalized:
        normalized["typical_p"] = normalized.get("typical")
    supported = {
        "model_id",
        "max_new_tokens",
        "seed",
        "stream",
        "temperature",
        "top_p",
        "top_k",
        "stop_strings",
        "stop_token_ids",
        "ignore_eos",
        "min_tokens",
        "best_of",
        "use_beam_search",
        "length_penalty",
        "include_stop_str_in_output",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "truncate_prompt_tokens",
        "allowed_token_ids",
        "prompt_logprobs",
        "system",
        "system_file",
        "user_prefix",
        "show_thinking",
        "show_tool_activity",
        "show_tool_arguments",
        "no_animate_thinking",
        "save_transcript",
        "history_strip_think",
        "tools_enabled",
        "tools_mode",
        "tools_schema_file",
        "tools_tool_choice",
        "tools_allow",
        "tools_deny",
        "tools_max_calls_per_turn",
        "tools_timeout_s",
        "tools_max_result_chars",
        "assume_think",
        "scroll_lines",
        "ui_tick_ms",
        "ui_max_events_per_tick",
        "capture_last_request",
        "telemetry_jsonl",
        "telemetry_sample_interval_s",
        "dtype",
        "hf_attn_implementation",
        "hf_log_file",
        "hf_device_map",
        "hf_max_memory",
        "hf_text_only",
        "hf_low_cpu_mem_usage",
        "prompt_mode",
        "chat_template",
        "max_context_tokens",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "typical_p",
        "min_p",
        "max_time",
        "num_beams",
        "no_repeat_ngram_size",
        "use_8bit",
        "use_4bit",
        "n_ctx",
        "n_gpu_layers",
        "model_path",
        "gguf_log_file",
        "max_seq_len",
        "min_free_tokens",
        "gpu_split",
        "cache_type",
        "exl2_stop_tokens",
        "exl2_repeat_streak_max",
        "rope_scale",
        "rope_alpha",
        "rope_yarn",
        "low_mem",
        "exl2_repo_path",
        "exl2_log_file",
        "flash_attn",
        "xformers",
        "sdpa",
        "graphs",
        "ollama_host",
        "ollama_timeout",
        "ollama_think",
        "ollama_log_file",
        "openai_base_url",
        "openai_api_key",
        "openai_timeout_s",
        "openai_log_file",
        "vllm_mode",
        "vllm_host",
        "vllm_port",
        "vllm_cmd",
        "vllm_extra_args",
        "vllm_served_model_name",
        "vllm_tensor_parallel_size",
        "vllm_gpu_memory_utilization",
        "vllm_max_model_len",
        "vllm_base_url",
        "vllm_api_key",
        "vllm_timeout_s",
        "vllm_generation_config",
        "vllm_attention_backend",
        "vllm_dtype",
        "vllm_enable_auto_tool_choice",
        "vllm_tool_call_parser",
        "vllm_log_file",
    }
    return {k: v for k, v in normalized.items() if k in supported}


def _detect_cli_overrides(argv: list[str]) -> set[str]:
    option_to_dest = {
        "--max-new-tokens": "max_new_tokens",
        "--seed": "seed",
        "--temperature": "temperature",
        "--top-p": "top_p",
        "--top-k": "top_k",
        "--stop-strings": "stop_strings",
        "--stop-token-ids": "stop_token_ids",
        "--ignore-eos": "ignore_eos",
        "--no-ignore-eos": "ignore_eos",
        "--min-tokens": "min_tokens",
        "--best-of": "best_of",
        "--use-beam-search": "use_beam_search",
        "--no-use-beam-search": "use_beam_search",
        "--length-penalty": "length_penalty",
        "--include-stop-str-in-output": "include_stop_str_in_output",
        "--no-include-stop-str-in-output": "include_stop_str_in_output",
        "--skip-special-tokens": "skip_special_tokens",
        "--no-skip-special-tokens": "skip_special_tokens",
        "--spaces-between-special-tokens": "spaces_between_special_tokens",
        "--no-spaces-between-special-tokens": "spaces_between_special_tokens",
        "--truncate-prompt-tokens": "truncate_prompt_tokens",
        "--allowed-token-ids": "allowed_token_ids",
        "--prompt-logprobs": "prompt_logprobs",
        "--system": "system",
        "--system-file": "system_file",
        "--user-prefix": "user_prefix",
        "--show-tool-activity": "show_tool_activity",
        "--no-show-tool-activity": "show_tool_activity",
        "--show-tool-arguments": "show_tool_arguments",
        "--no-show-tool-arguments": "show_tool_arguments",
        "--assume-think": "assume_think",
        "--no-assume-think": "assume_think",
        "--capture-last-request": "capture_last_request",
        "--telemetry-jsonl": "telemetry_jsonl",
        "--telemetry-sample-interval-s": "telemetry_sample_interval_s",
        "--tools-enabled": "tools_enabled",
        "--no-tools-enabled": "tools_enabled",
        "--tools-mode": "tools_mode",
        "--tools-schema-file": "tools_schema_file",
        "--tools-tool-choice": "tools_tool_choice",
        "--tools-allow": "tools_allow",
        "--tools-deny": "tools_deny",
        "--tools-max-calls-per-turn": "tools_max_calls_per_turn",
        "--tools-timeout-s": "tools_timeout_s",
        "--tools-max-result-chars": "tools_max_result_chars",
        "--model-path": "model_path",
        "--hf-log-file": "hf_log_file",
        "--gguf-log-file": "gguf_log_file",
        "--exl2-log-file": "exl2_log_file",
        "--ollama-host": "ollama_host",
        "--ollama-timeout": "ollama_timeout",
        "--ollama-think": "ollama_think",
        "--ollama-log-file": "ollama_log_file",
        "--openai-base-url": "openai_base_url",
        "--openai-api-key": "openai_api_key",
        "--openai-timeout-s": "openai_timeout_s",
        "--openai-log-file": "openai_log_file",
        "--vllm-mode": "vllm_mode",
        "--vllm-host": "vllm_host",
        "--vllm-port": "vllm_port",
        "--vllm-cmd": "vllm_cmd",
        "--vllm-extra-args": "vllm_extra_args",
        "--vllm-served-model-name": "vllm_served_model_name",
        "--vllm-tensor-parallel-size": "vllm_tensor_parallel_size",
        "--vllm-gpu-memory-utilization": "vllm_gpu_memory_utilization",
        "--vllm-max-model-len": "vllm_max_model_len",
        "--vllm-base-url": "vllm_base_url",
        "--vllm-api-key": "vllm_api_key",
        "--vllm-timeout-s": "vllm_timeout_s",
        "--vllm-generation-config": "vllm_generation_config",
        "--vllm-attention-backend": "vllm_attention_backend",
        "--vllm-dtype": "vllm_dtype",
        "--vllm-enable-auto-tool-choice": "vllm_enable_auto_tool_choice",
        "--no-vllm-enable-auto-tool-choice": "vllm_enable_auto_tool_choice",
        "--vllm-tool-call-parser": "vllm_tool_call_parser",
        "--vllm-log-file": "vllm_log_file",
    }
    seen: set[str] = set()
    for token in argv:
        if token == "--":
            break
        if not token.startswith("--"):
            continue
        key = token.split("=", 1)[0]
        dest = option_to_dest.get(key)
        if dest:
            seen.add(dest)
    return seen


def _warn_ignored_flags(parser: argparse.ArgumentParser, args: argparse.Namespace, backend: str):
    backend_relevant = {
        "hf": {
            "dtype",
            "hf_attn_implementation",
            "hf_log_file",
            "hf_device_map",
            "hf_max_memory",
            "hf_text_only",
            "hf_low_cpu_mem_usage",
            "prompt_mode",
            "chat_template",
            "max_context_tokens",
            "repetition_penalty",
            "typical_p",
            "min_p",
            "max_time",
            "num_beams",
            "no_repeat_ngram_size",
            "use_8bit",
            "use_4bit",
        },
        "gguf": {
            "n_ctx",
            "n_gpu_layers",
            "gguf_log_file",
            "prompt_mode",
            "chat_template",
            "top_k",
            "min_p",
            "typical_p",
            "repetition_penalty",
            "model_path",
        },
        "ollama": {"ollama_host", "ollama_timeout", "ollama_think", "ollama_log_file"},
        "openai": {"openai_base_url", "openai_api_key", "openai_timeout_s", "openai_log_file"},
        "vllm": {
            "vllm_mode",
            "vllm_host",
            "vllm_port",
            "vllm_cmd",
            "vllm_extra_args",
            "vllm_served_model_name",
            "vllm_tensor_parallel_size",
            "vllm_gpu_memory_utilization",
            "vllm_max_model_len",
            "vllm_base_url",
            "vllm_api_key",
            "vllm_timeout_s",
            "vllm_generation_config",
            "vllm_attention_backend",
            "vllm_dtype",
            "vllm_enable_auto_tool_choice",
            "vllm_tool_call_parser",
            "vllm_log_file",
            "top_k",
            "min_p",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "stop_token_ids",
            "ignore_eos",
            "min_tokens",
            "best_of",
            "use_beam_search",
            "length_penalty",
            "include_stop_str_in_output",
            "skip_special_tokens",
            "spaces_between_special_tokens",
            "truncate_prompt_tokens",
            "allowed_token_ids",
            "prompt_logprobs",
        },
        "exl2": {
            "model_path",
            "prompt_mode",
            "chat_template",
            "top_k",
            "min_p",
            "typical_p",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "max_seq_len",
            "min_free_tokens",
            "gpu_split",
            "cache_type",
            "exl2_stop_tokens",
            "exl2_repeat_streak_max",
            "rope_scale",
            "rope_alpha",
            "rope_yarn",
            "low_mem",
            "exl2_repo_path",
            "exl2_log_file",
            "flash_attn",
            "xformers",
            "sdpa",
            "graphs",
        },
    }

    common = {
        "model_id",
        "backend",
        "config",
        "profile",
        "backend_only",
        "detach_backend",
        "shutdown_backend",
        "max_new_tokens",
        "seed",
        "stream",
        "temperature",
        "top_p",
        "top_k",
        "stop_strings",
        "stop_token_ids",
        "ignore_eos",
        "min_tokens",
        "best_of",
        "use_beam_search",
        "length_penalty",
        "include_stop_str_in_output",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "truncate_prompt_tokens",
        "allowed_token_ids",
        "prompt_logprobs",
        "system",
        "system_file",
        "user_prefix",
        "show_thinking",
        "no_animate_thinking",
        "save_transcript",
        "assume_think",
        "history_strip_think",
        "scroll_lines",
        "ui_tick_ms",
        "ui_max_events_per_tick",
        "capture_last_request",
        "telemetry_jsonl",
        "telemetry_sample_interval_s",
    }

    used = common | backend_relevant[backend]
    ignored = []
    for action in parser._actions:
        if not action.option_strings:
            continue
        dest = action.dest
        if dest in used:
            continue
        if not hasattr(args, dest):
            continue
        value = getattr(args, dest)
        default = parser.get_default(dest)
        if value != default:
            ignored.append(f"--{dest.replace('_', '-')}")

    if ignored:
        print(f"Ignored for backend '{backend}': {', '.join(sorted(set(ignored)))}")


def parse_args() -> argparse.Namespace:
    raw_argv = sys.argv[1:]
    cli_overrides = _detect_cli_overrides(raw_argv)

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("model_id", nargs="?")
    pre.add_argument("backend_hint", nargs="?", choices=["hf", "gguf", "ollama", "exl2", "openai", "vllm"])
    pre.add_argument("--backend", choices=["hf", "gguf", "ollama", "exl2", "openai", "vllm"], default=None)
    pre.add_argument("--config", default="")
    pre.add_argument("--profile", default="")
    pre_args, _ = pre.parse_known_args()

    backend_override = pre_args.backend or pre_args.backend_hint
    backend = detect_backend(pre_args.model_id, backend_override)
    if pre_args.model_id and not pre_args.config and pre_args.backend is None and pre_args.backend_hint is None:
        inferred = _infer_backend_from_default_config(pre_args.model_id)
        if inferred:
            backend = inferred

    config_data = {}
    config_path = None
    config_meta = {}
    if pre_args.config:
        config_data, config_meta = load_config_layers(
            pre_args.config,
            backend=backend,
            profile=pre_args.profile,
            include_machine=True,
        )
        config_path = config_meta.get("base")
        print(f"Loaded config: {config_path}")
    elif pre_args.model_id:
        config_data, config_path, config_meta = _load_default_backend_config(
            pre_args.model_id,
            backend=backend,
            profile=pre_args.profile,
        )
        if config_path:
            print(f"Loaded default config: {config_path}")
    if config_meta is None:
        config_meta = {}

    parser = build_parser()
    defaults = _collect_config_defaults(config_data)
    if defaults:
        parser.set_defaults(**defaults)

    args = parser.parse_args()
    args._cli_overrides = cli_overrides
    args._config_keys = set(defaults.keys())
    args._config_profile = pre_args.profile or ""
    args._config_layers = list(config_meta.get("loaded", []) or ([] if not config_path else [config_path]))
    args._config_origins = dict(config_meta.get("origins", {}) or {})
    args.display_name = str(config_data.get("display_name", "") or "") if isinstance(config_data, dict) else ""
    early_backend = detect_backend(args.model_id, args.backend)
    if early_backend == "openai" and not pre_args.model_id:
        # Attach mode should not inherit a local model path from config defaults.
        # Let the OpenAI-compatible session resolve /v1/models unless the user
        # explicitly provided a model id.
        args.model_id = ""
    if not args.model_id and early_backend == "gguf":
        args.model_id = defaults.get("model_path")
    if early_backend == "gguf":
        args.model_id = _resolve_gguf_model_id(
            args.model_id,
            model_path=args.model_path,
            config_path=config_path,
        )
    if not args.model_id and early_backend != "openai":
        args.model_id = defaults.get("model_id")
    if not args.model_id and early_backend != "openai":
        parser.error("model_id is required (or provide model_id in --config).")
    if not args.model_id and early_backend == "openai":
        args.model_id = ""

    backend_override = args.backend or args.backend_hint
    args.backend = detect_backend(args.model_id, backend_override)
    if args.model_id and not pre_args.config and pre_args.backend is None and pre_args.backend_hint is None:
        inferred = _infer_backend_from_default_config(args.model_id)
        if inferred:
            args.backend = inferred
    args._config_path = config_path
    if args.assume_think is None:
        args.assume_think = False
    if args.history_strip_think is None:
        args.history_strip_think = False
    if args.backend == "gguf":
        args.model_id = _resolve_gguf_model_id(
            args.model_id,
            model_path=args.model_path,
            config_path=config_path,
        )
    if args.backend == "exl2" and isinstance(args.model_id, str) and args.model_id.startswith("exl2:"):
        args.model_id = args.model_id.split(":", 1)[1]
    if args.backend == "openai" and isinstance(args.model_id, str) and args.model_id.startswith("openai:"):
        args.model_id = args.model_id.split(":", 1)[1]
    if args.backend == "vllm" and isinstance(args.model_id, str) and args.model_id.startswith("vllm:"):
        args.model_id = args.model_id.split(":", 1)[1]
    if args.backend not in {"openai", "ollama"} and isinstance(args.model_id, str):
        args.model_id = apply_machine_model_root(args.model_id)
    if args.backend == "openai" and not pre_args.model_id:
        args.model_id = ""
    if pre_args.model_id:
        args._config_origins["model_id"] = "cli(positional)"
    for key in cli_overrides:
        args._config_origins[key] = "cli"

    if not args.system and args.system_file:
        try:
            path = _resolve_path_maybe_relative(args.system_file, config_path=config_path)
            with open(path, "r", encoding="utf-8") as fh:
                args.system = fh.read().strip()
        except Exception as exc:
            print(f"Failed to read system file '{args.system_file}': {exc}")
            sys.exit(1)

    _warn_ignored_flags(parser, args, args.backend)
    return args


def _mirror_engine_logs_to_stdout(
    stdout_path: str,
    stderr_path: str,
    *,
    stdout_start: int = 0,
    stderr_start: int = 0,
) -> tuple[threading.Event, list[threading.Thread]]:
    stop_event = threading.Event()
    threads: list[threading.Thread] = []

    def _tail(path: str, target, start_offset: int) -> None:
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                fh.seek(max(0, int(start_offset)))
                while not stop_event.is_set():
                    line = fh.readline()
                    if line:
                        print(line.rstrip("\n"), file=target, flush=True)
                        continue
                    time.sleep(0.15)
        except Exception:
            return

    if stdout_path:
        t = threading.Thread(target=_tail, args=(stdout_path, sys.stdout, stdout_start), daemon=True)
        t.start()
        threads.append(t)
    if stderr_path:
        t = threading.Thread(target=_tail, args=(stderr_path, sys.stderr, stderr_start), daemon=True)
        t.start()
        threads.append(t)
    return stop_event, threads


def main() -> None:
    args = parse_args()
    session = None
    telemetry = None

    if args.shutdown_backend:
        if args.backend != "vllm":
            print("--shutdown-backend is only supported for the managed vLLM backend.")
            sys.exit(1)
        try:
            result = shutdown_managed_server(args)
        except Exception as exc:
            print(f"Failed to shut down managed vLLM backend: {exc}")
            sys.exit(1)
        if result.get("stopped"):
            print(
                "Managed vLLM stopped: "
                f"pid={result.get('pid')} base_url={result.get('base_url')} control_file={result.get('control_file')}"
            )
            return
        print(f"No managed vLLM stopped ({result.get('reason')}): control_file={result.get('control_file')}")
        return

    try:
        if args.backend == "hf":
            session = create_hf_session(args)
        elif args.backend == "gguf":
            session = create_gguf_session(args)
        elif args.backend == "ollama":
            session = create_ollama_session(args)
        elif args.backend == "exl2":
            session = create_exl2_session(args)
        elif args.backend == "openai":
            session = create_openai_session(args)
        elif args.backend == "vllm":
            session = create_vllm_session(args)
        else:
            raise RuntimeError(f"Unknown backend: {args.backend}")
    except Exception as exc:
        print(f"Failed to initialize backend '{args.backend}': {exc}")
        print("If tokenizer conversion fails, install: sentencepiece, tiktoken, protobuf")
        sys.exit(1)

    telemetry = TelemetryContext.create(args, session)
    if telemetry.enabled:
        attach_log_subscribers(session, lambda source, line: telemetry.publish_log_record(source=source, message=line))
        telemetry.publish_session_started()
        telemetry.publish_load_report(session, args)

    if args.backend_only:
        if args.backend != "vllm":
            print("--backend-only is currently supported only for the managed vLLM backend.")
            closer = getattr(session, "close", None)
            if callable(closer):
                closer()
            sys.exit(1)
        describe = getattr(session, "describe", None)
        info = describe() if callable(describe) else {}
        if not isinstance(info, dict):
            info = {}
        print(
            "Backend ready "
            f"(backend={args.backend}, base_url={info.get('base_url')}, pid={info.get('pid')}, "
            f"control_file={info.get('control_file')})"
        )
        if args.detach_backend:
            detacher = getattr(session, "detach", None)
            if callable(detacher):
                detacher()
            closer = getattr(session, "close", None)
            if callable(closer):
                closer()
            return
        print("Serving in foreground. Press Ctrl+C to stop.")
        mirror_stop = None
        mirror_threads: list[threading.Thread] = []
        stdout_path = str(info.get("engine_stdout_path") or "")
        stderr_path = str(info.get("engine_stderr_path") or "")
        if stdout_path or stderr_path:
            mirror_stop, mirror_threads = _mirror_engine_logs_to_stdout(
                stdout_path,
                stderr_path,
                stdout_start=int(info.get("engine_stdout_start") or 0),
                stderr_start=int(info.get("engine_stderr_start") or 0),
            )
        try:
            while True:
                signal.pause()
        except KeyboardInterrupt:
            pass
        finally:
            if telemetry is not None and telemetry.enabled:
                telemetry.publish_session_finished(status="finished")
                telemetry.close()
            if mirror_stop is not None:
                mirror_stop.set()
            for thread in mirror_threads:
                try:
                    thread.join(timeout=0.5)
                except Exception:
                    pass
            closer = getattr(session, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception as exc:
                    print(f"Warning: backend shutdown failed: {exc}")
        return

    print(f"TUI ready (backend={args.backend}). Commands: /exit, /quit, /clear. Ctrl+Q exits TUI only.")
    app = UnifiedTuiApp(TuiRuntime(session=session, args=args, telemetry=telemetry))
    if args.detach_backend:
        app.shutdown_backend_on_exit = False
    try:
        app.run()
    finally:
        if telemetry is not None and telemetry.enabled:
            telemetry.close()
        if not getattr(app, "shutdown_backend_on_exit", True):
            detacher = getattr(session, "detach", None)
            if callable(detacher):
                try:
                    detacher()
                except Exception:
                    pass
        closer = getattr(session, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception as exc:
                print(f"Warning: backend shutdown failed: {exc}")


if __name__ == "__main__":
    main()
