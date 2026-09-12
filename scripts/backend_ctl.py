#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _python_bin() -> str:
    venv_python = ROOT / ".venv" / "bin" / "python"
    if venv_python.is_file() and os.access(venv_python, os.X_OK):
        return str(venv_python)
    return sys.executable or "python3"


def _build_base_argv() -> list[str]:
    return [_python_bin(), str(ROOT / "tui.py")]


def _venv_site_packages_dir() -> Path | None:
    lib_root = ROOT / ".venv" / "lib"
    if not lib_root.is_dir():
        return None

    candidates = sorted(lib_root.glob("python*/site-packages"))
    if not candidates:
        return None
    return candidates[-1]


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    # Helps reduce CUDA memory fragmentation under long-lived/large-model vLLM runs.
    # Respect explicit user overrides from the shell.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    site_packages = _venv_site_packages_dir()
    if site_packages is None:
        return env

    lib_dirs: list[str] = []

    torch_lib = site_packages / "torch" / "lib"
    if torch_lib.is_dir():
        lib_dirs.append(str(torch_lib))

    nvidia_root = site_packages / "nvidia"
    if nvidia_root.is_dir():
        for child in sorted(nvidia_root.iterdir()):
            lib_dir = child / "lib"
            if lib_dir.is_dir():
                lib_dirs.append(str(lib_dir))

    current = env.get("LD_LIBRARY_PATH", "")
    if current:
        lib_dirs.extend(part for part in current.split(":") if part)

    deduped: list[str] = []
    seen: set[str] = set()
    for path in lib_dirs:
        if path and path not in seen:
            deduped.append(path)
            seen.add(path)

    if deduped:
        env["LD_LIBRARY_PATH"] = ":".join(deduped)
    return env


def _has_flag(argv: list[str], flag: str) -> bool:
    return any(token == flag or token.startswith(flag + "=") for token in argv)


def _main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] in {"vllm-up", "vllm-up-bg", "vllm-down", "backend-up-bg", "backend-down-bg"}:
        tool_name = sys.argv[1]
        user_args = list(sys.argv[2:])
    else:
        tool_name = Path(sys.argv[0]).name
        user_args = list(sys.argv[1:])

    argv = _build_base_argv()
    if tool_name in {"vllm-up", "vllm-up-bg", "vllm-down"} and not _has_flag(user_args, "--backend"):
        argv.extend(["--backend", "vllm"])

    detached = False
    if tool_name == "vllm-up":
        filtered_args: list[str] = []
        for token in user_args:
            if token == "--bg":
                detached = True
                continue
            if token == "--fg":
                detached = False
                continue
            filtered_args.append(token)
        user_args = filtered_args

    if tool_name == "vllm-up":
        argv.append("--backend-only")
        if detached:
            argv.append("--detach-backend")
    elif tool_name == "vllm-up-bg":
        argv.extend(["--backend-only", "--detach-backend"])
    elif tool_name == "vllm-down":
        argv.append("--shutdown-backend")
    elif tool_name == "backend-up-bg":
        argv.extend(["--backend-only", "--detach-backend"])
    elif tool_name == "backend-down-bg":
        argv.append("--shutdown-backend")
    else:
        print(
            "Unsupported backend control command. "
            f"Expected one of: vllm-up, vllm-up-bg, vllm-down, backend-up-bg, backend-down-bg (got {tool_name})",
            file=sys.stderr,
        )
        return 2

    argv.extend(user_args)
    return subprocess.call(argv, cwd=str(ROOT), env=_runtime_env())


if __name__ == "__main__":
    raise SystemExit(_main())
