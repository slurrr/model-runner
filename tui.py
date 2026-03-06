from __future__ import annotations

import argparse
import os
import re
import sys

from config_utils import load_default_json_config_for_model, load_json_config
from tui_app.app import TuiRuntime, UnifiedTuiApp
from tui_app.backends.exl2 import create_session as create_exl2_session
from tui_app.backends.gguf import create_session as create_gguf_session
from tui_app.backends.hf import create_session as create_hf_session
from tui_app.backends.ollama import create_session as create_ollama_session


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


def _load_default_backend_config(model: str, backend: str):
    if backend != "ollama":
        return load_default_json_config_for_model(model, backend=backend)

    candidates = [model]
    if model.startswith("ollama:"):
        candidates.append(model.split(":", 1)[1])
    candidates.append(sanitize_ollama_stem(model))

    for candidate in candidates:
        data, path = load_default_json_config_for_model(candidate, backend=backend)
        if path:
            return data, path
    return None, None


def _infer_backend_from_default_config(model: str) -> str | None:
    matches: list[str] = []
    for backend in ("hf", "gguf", "ollama", "exl2"):
        _, path = _load_default_backend_config(model, backend=backend)
        if path:
            matches.append(backend)
    if len(matches) == 1:
        return matches[0]
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Textual TUI for HF + GGUF + Ollama + EXL2 backends")
    parser.add_argument("model_id", nargs="?", help="Model id/path, .gguf path, or ollama:<name>")
    parser.add_argument("--backend", choices=["hf", "gguf", "ollama", "exl2"], default=None)
    parser.add_argument("--config", default="", help="Config path or name.")

    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stream", dest="stream", action="store_true")
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--stop-strings", nargs="+", default=None)

    parser.add_argument("--system", default="")
    parser.add_argument("--system-file", default="")
    parser.add_argument("--user-prefix", default="")

    parser.add_argument("--show-thinking", action="store_true")
    parser.add_argument("--no-animate-thinking", action="store_true")
    parser.add_argument("--save-transcript", default="")
    think_mode_group = parser.add_mutually_exclusive_group()
    think_mode_group.add_argument("--assume-think", dest="assume_think", action="store_true")
    think_mode_group.add_argument("--no-assume-think", dest="assume_think", action="store_false")

    parser.add_argument("--scroll-lines", type=int, default=1)
    parser.add_argument("--ui-tick-ms", type=int, default=33)
    parser.add_argument("--ui-max-events-per-tick", type=int, default=120)

    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--prompt-mode", choices=["chat", "plain"], default="chat")
    parser.add_argument("--chat-template", default="")
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

    parser.set_defaults(stream=True, assume_think=None)
    parser.set_defaults(flash_attn=None, xformers=None, sdpa=None, graphs=None)
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
        "system",
        "system_file",
        "user_prefix",
        "show_thinking",
        "no_animate_thinking",
        "save_transcript",
        "assume_think",
        "scroll_lines",
        "ui_tick_ms",
        "ui_max_events_per_tick",
        "dtype",
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
        "flash_attn",
        "xformers",
        "sdpa",
        "graphs",
        "ollama_host",
        "ollama_timeout",
        "ollama_think",
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
        "--system": "system",
        "--system-file": "system_file",
        "--user-prefix": "user_prefix",
        "--assume-think": "assume_think",
        "--no-assume-think": "assume_think",
        "--model-path": "model_path",
        "--ollama-host": "ollama_host",
        "--ollama-timeout": "ollama_timeout",
        "--ollama-think": "ollama_think",
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
            "prompt_mode",
            "chat_template",
            "top_k",
            "min_p",
            "typical_p",
            "repetition_penalty",
            "model_path",
        },
        "ollama": {"ollama_host", "ollama_timeout", "ollama_think"},
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
        "max_new_tokens",
        "seed",
        "stream",
        "temperature",
        "top_p",
        "top_k",
        "stop_strings",
        "system",
        "system_file",
        "user_prefix",
        "show_thinking",
        "no_animate_thinking",
        "save_transcript",
        "assume_think",
        "scroll_lines",
        "ui_tick_ms",
        "ui_max_events_per_tick",
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
    pre.add_argument("--backend", choices=["hf", "gguf", "ollama", "exl2"], default=None)
    pre.add_argument("--config", default="")
    pre_args, _ = pre.parse_known_args()

    backend = detect_backend(pre_args.model_id, pre_args.backend)
    if pre_args.model_id and not pre_args.config and pre_args.backend is None:
        inferred = _infer_backend_from_default_config(pre_args.model_id)
        if inferred:
            backend = inferred

    config_data = {}
    config_path = None
    if pre_args.config:
        config_data, config_path = load_json_config(pre_args.config, backend=backend)
        print(f"Loaded config: {config_path}")
    elif pre_args.model_id:
        config_data, config_path = _load_default_backend_config(pre_args.model_id, backend=backend)
        if config_path:
            print(f"Loaded default config: {config_path}")

    parser = build_parser()
    defaults = _collect_config_defaults(config_data)
    if defaults:
        parser.set_defaults(**defaults)

    args = parser.parse_args()
    args._cli_overrides = cli_overrides
    args._config_keys = set(defaults.keys())
    early_backend = detect_backend(args.model_id, args.backend)
    if early_backend == "gguf" and args.model_path and not args.model_id:
        args.model_id = _resolve_path_maybe_relative(args.model_path, config_path=config_path)
    if not args.model_id:
        args.model_id = defaults.get("model_id")
    if not args.model_id:
        parser.error("model_id is required (or provide model_id in --config).")

    args.backend = detect_backend(args.model_id, args.backend)
    if args.model_id and not pre_args.config and pre_args.backend is None:
        inferred = _infer_backend_from_default_config(args.model_id)
        if inferred:
            args.backend = inferred
    args._config_path = config_path
    if args.assume_think is None:
        args.assume_think = False
    if args.backend == "gguf" and args.model_path and not args.model_id:
        args.model_id = _resolve_path_maybe_relative(args.model_path, config_path=config_path)
    if args.backend == "exl2" and isinstance(args.model_id, str) and args.model_id.startswith("exl2:"):
        args.model_id = args.model_id.split(":", 1)[1]

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


def main() -> None:
    args = parse_args()

    try:
        if args.backend == "hf":
            session = create_hf_session(args)
        elif args.backend == "gguf":
            session = create_gguf_session(args)
        elif args.backend == "ollama":
            session = create_ollama_session(args)
        elif args.backend == "exl2":
            session = create_exl2_session(args)
        else:
            raise RuntimeError(f"Unknown backend: {args.backend}")
    except Exception as exc:
        print(f"Failed to initialize backend '{args.backend}': {exc}")
        print("If tokenizer conversion fails, install: sentencepiece, tiktoken, protobuf")
        sys.exit(1)

    print(f"TUI ready (backend={args.backend}). Commands: /exit, /quit, /clear")
    app = UnifiedTuiApp(TuiRuntime(session=session, args=args))
    app.run()


if __name__ == "__main__":
    main()
