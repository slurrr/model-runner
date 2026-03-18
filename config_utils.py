from __future__ import annotations

import json
import os
import re
import tomllib


def _repo_root() -> str:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.path.abspath(os.getcwd())
    if (
        os.path.isdir(os.path.join(cwd, "models"))
        and os.path.isdir(os.path.join(cwd, "docs"))
        and os.path.isfile(os.path.join(cwd, "tui.py"))
    ):
        return cwd
    return module_dir


def _load_raw_config(path: str) -> dict:
    with open(path, "rb") as fh:
        if path.endswith(".toml"):
            data = tomllib.load(fh)
        else:
            data = json.loads(fh.read().decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be an object at top level: {path}")
    return data


def _flatten_toml_config(data: dict, *, backend: str) -> dict:
    out: dict = {}

    model = data.get("model")
    if isinstance(model, dict):
        if "id" in model:
            out["model_id"] = model.get("id")
        if "display_name" in model:
            out["display_name"] = model.get("display_name")

    for section in ("gen", "prompt", "ui", "tools", "telemetry"):
        sec = data.get(section)
        if isinstance(sec, dict):
            for key, value in sec.items():
                mapped = key
                if section == "tools":
                    mapped = f"tools_{key}"
                elif section == "telemetry":
                    mapped = f"telemetry_{key}"
                out[mapped] = value

    backend_root = data.get("backend")
    if isinstance(backend_root, dict):
        bsec = backend_root.get(backend)
        if isinstance(bsec, dict):
            for key, value in bsec.items():
                mapped = key
                if backend == "ollama":
                    if key == "host":
                        mapped = "ollama_host"
                    elif key == "timeout":
                        mapped = "ollama_timeout"
                    elif key == "think_mode":
                        mapped = "ollama_think"
                elif backend == "hf" and key == "attn_implementation":
                    mapped = "hf_attn_implementation"
                elif backend == "hf" and key == "log_file":
                    mapped = "hf_log_file"
                elif backend == "hf" and key == "device_map":
                    mapped = "hf_device_map"
                elif backend == "hf" and key == "max_memory":
                    mapped = "hf_max_memory"
                elif backend == "hf" and key == "text_only":
                    mapped = "hf_text_only"
                elif backend == "hf" and key == "low_cpu_mem_usage":
                    mapped = "hf_low_cpu_mem_usage"
                elif backend == "gguf" and key == "log_file":
                    mapped = "gguf_log_file"
                elif backend == "exl2" and key == "log_file":
                    mapped = "exl2_log_file"
                elif backend == "openai":
                    if key == "base_url":
                        mapped = "openai_base_url"
                    elif key == "api_key":
                        mapped = "openai_api_key"
                    elif key == "timeout_s":
                        mapped = "openai_timeout_s"
                    elif key == "log_file":
                        mapped = "openai_log_file"
                elif backend == "ollama" and key == "log_file":
                    mapped = "ollama_log_file"
                elif backend == "vllm":
                    if key == "host":
                        mapped = "vllm_host"
                    elif key == "port":
                        mapped = "vllm_port"
                    elif key == "mode":
                        mapped = "vllm_mode"
                    elif key == "cmd":
                        mapped = "vllm_cmd"
                    elif key == "extra_args":
                        mapped = "vllm_extra_args"
                    elif key == "served_model_name":
                        mapped = "vllm_served_model_name"
                    elif key == "tensor_parallel_size":
                        mapped = "vllm_tensor_parallel_size"
                    elif key == "gpu_memory_utilization":
                        mapped = "vllm_gpu_memory_utilization"
                    elif key == "max_model_len":
                        mapped = "vllm_max_model_len"
                    elif key == "base_url":
                        mapped = "vllm_base_url"
                    elif key == "api_key":
                        mapped = "vllm_api_key"
                    elif key == "timeout_s":
                        mapped = "vllm_timeout_s"
                    elif key == "generation_config":
                        mapped = "vllm_generation_config"
                    elif key == "attention_backend":
                        mapped = "vllm_attention_backend"
                    elif key == "dtype":
                        mapped = "vllm_dtype"
                    elif key == "enable_auto_tool_choice":
                        mapped = "vllm_enable_auto_tool_choice"
                    elif key == "tool_call_parser":
                        mapped = "vllm_tool_call_parser"
                    elif key == "log_file":
                        mapped = "vllm_log_file"
                out[mapped] = value

    return out


def _read_config_as_flat(path: str, *, backend: str) -> dict:
    data = _load_raw_config(path)
    if path.endswith(".toml"):
        return _flatten_toml_config(data, backend=backend)
    return data


def _merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _merge_layers_with_origins(layers: list[tuple[str, dict]]) -> tuple[dict, dict]:
    merged: dict = {}
    origins: dict = {}
    for source, data in layers:
        for key, value in data.items():
            merged[key] = value
            origins[key] = source
    return merged, origins


def _is_local_stem(model_id: str) -> bool:
    value = (model_id or "").strip()
    if not value:
        return False
    lowered = value.lower()
    if lowered.startswith(("ollama:", "exl2:")):
        return False
    if value.startswith((".", "~")):
        return False
    if "/" in value or "\\" in value:
        return False
    if ":" in value:
        return False
    return True


def _apply_machine_overrides(data: dict, *, backend: str, machine_path: str) -> dict:
    if not os.path.isfile(machine_path):
        return dict(data)
    raw = _load_raw_config(machine_path)
    machine = raw.get("machine")
    if not isinstance(machine, dict):
        return dict(data)

    out = dict(data)
    model_root = machine.get("model_root")
    gguf_model_root = machine.get("gguf_model_root")
    exl2_model_root = machine.get("exl2_model_root")
    exl2_repo_path = machine.get("exl2_repo_path")

    def _clean_str(value: object) -> str:
        return value.strip() if isinstance(value, str) else ""

    def _pathish_is_relative(value: str) -> bool:
        if not value:
            return False
        expanded = os.path.expanduser(value)
        if os.path.isabs(expanded):
            return False
        if re.match(r"^[A-Za-z]:[\\/]", value):
            return False
        return True

    if isinstance(model_root, str) and model_root.strip() and isinstance(out.get("model_id"), str):
        model_id = out["model_id"].strip()
        if _is_local_stem(model_id):
            out["model_id"] = os.path.join(os.path.expanduser(model_root), model_id)

    if backend == "gguf":
        gguf_root = _clean_str(gguf_model_root) or _clean_str(model_root)
        raw_model_path = _clean_str(out.get("model_path"))
        if gguf_root and _pathish_is_relative(raw_model_path):
            out["model_path"] = os.path.join(os.path.expanduser(gguf_root), raw_model_path)

    if backend == "exl2":
        exl2_root = _clean_str(exl2_model_root) or _clean_str(model_root)
        raw_model_path = _clean_str(out.get("model_path"))
        raw_model_id = _clean_str(out.get("model_id"))
        if exl2_root and not raw_model_path and _is_local_stem(raw_model_id):
            out["model_path"] = os.path.expanduser(exl2_root)
        elif exl2_root and _pathish_is_relative(raw_model_path):
            out["model_path"] = os.path.join(os.path.expanduser(exl2_root), raw_model_path)

        raw_repo_path = _clean_str(out.get("exl2_repo_path"))
        if _clean_str(exl2_repo_path) and not raw_repo_path:
            out["exl2_repo_path"] = _clean_str(exl2_repo_path)
        elif _clean_str(exl2_repo_path) and _pathish_is_relative(raw_repo_path):
            out["exl2_repo_path"] = os.path.join(os.path.expanduser(_clean_str(exl2_repo_path)), raw_repo_path)

    if backend == "ollama":
        ollama_host = machine.get("ollama_host")
        if isinstance(ollama_host, str) and ollama_host.strip():
            out["ollama_host"] = ollama_host.strip()
    hf_cache_dir = machine.get("hf_cache_dir")
    if isinstance(hf_cache_dir, str) and hf_cache_dir.strip():
        if not out.get("hf_cache_dir"):
            out["hf_cache_dir"] = hf_cache_dir.strip()
    openai_base_url = machine.get("openai_base_url")
    if isinstance(openai_base_url, str) and openai_base_url.strip():
        if not out.get("openai_base_url"):
            out["openai_base_url"] = openai_base_url.strip()
        if backend == "vllm" and not out.get("vllm_base_url"):
            out["vllm_base_url"] = openai_base_url.strip()
    return out


def _machine_override_updates(data: dict, *, backend: str, machine_path: str) -> dict:
    updated = _apply_machine_overrides(data, backend=backend, machine_path=machine_path)
    out: dict = {}
    for key, value in updated.items():
        if data.get(key) != value:
            out[key] = value
    return out


def apply_machine_model_root(model_id: str) -> str:
    value = (model_id or "").strip()
    if not value:
        return value
    machine_path = os.path.join(_repo_root(), "config", "machine.toml")
    if not os.path.isfile(machine_path):
        return model_id
    raw = _load_raw_config(machine_path)
    machine = raw.get("machine")
    if not isinstance(machine, dict):
        return model_id
    model_root = machine.get("model_root")
    if not isinstance(model_root, str) or not model_root.strip():
        return model_id
    if not _is_local_stem(value):
        return model_id
    return os.path.join(os.path.expanduser(model_root), value)


def _resolve_profile_path(base_path: str, profile: str) -> str:
    if not profile:
        return ""
    base_dir = os.path.dirname(base_path)
    name = profile if profile.endswith(".toml") else f"{profile}.toml"
    return os.path.join(base_dir, "profiles", name)


def resolve_config_path(config_arg: str, backend: str = "hf") -> str | None:
    if not config_arg:
        return None

    raw = os.path.expanduser(config_arg.strip())
    root = _repo_root()
    has_ext = raw.endswith(".json") or raw.endswith(".toml")

    candidates = [raw]
    if not has_ext:
        candidates.append(f"{raw}.toml")
        candidates.append(f"{raw}.json")

    # Relative to repo root.
    candidates.append(os.path.join(root, raw))
    if not has_ext:
        candidates.append(os.path.join(root, f"{raw}.toml"))
        candidates.append(os.path.join(root, f"{raw}.json"))

    basename = os.path.basename(raw)
    stem = basename
    for suffix in (".json", ".toml"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    if stem == "template":
        stem = "_TEMPLATE"

    # Model-first layout (preferred).
    candidates.append(os.path.join(root, "models", stem, backend, "config", "default.toml"))
    candidates.append(os.path.join(root, "models", stem, backend, "config", "config.json"))
    candidates.append(os.path.join(root, "models", stem, backend, "config"))
    candidates.append(os.path.join(root, "models", stem, backend))
    candidates.append(os.path.join(root, "models", stem))

    # Backend-scoped legacy fallbacks.
    candidates.append(os.path.join(root, "config", backend, basename))
    if not has_ext:
        candidates.append(os.path.join(root, "config", backend, f"{stem}.toml"))
        candidates.append(os.path.join(root, "config", backend, f"{stem}.json"))
    candidates.append(os.path.join(root, "config", backend, stem, "default.toml"))
    candidates.append(os.path.join(root, "config", backend, stem, "config.json"))
    candidates.append(os.path.join(root, "config", backend, stem))

    seen = set()
    for path in candidates:
        normalized = os.path.abspath(path)
        if normalized in seen:
            continue
        seen.add(normalized)

        if os.path.isfile(normalized):
            return normalized

        if os.path.isdir(normalized):
            inside_toml = os.path.join(normalized, "default.toml")
            if os.path.isfile(inside_toml):
                return inside_toml
            inside_json = os.path.join(normalized, "config.json")
            if os.path.isfile(inside_json):
                return inside_json

    # Allow the OpenAI-compatible attach path to reuse an existing managed-vLLM
    # model config when there is no dedicated openai config folder.
    if backend == "openai":
        vllm_backend = "vllm"
        vllm_candidates = []
        vllm_candidates.append(os.path.join(root, "models", stem, vllm_backend, "config", "default.toml"))
        vllm_candidates.append(os.path.join(root, "models", stem, vllm_backend, "config", "config.json"))
        vllm_candidates.append(os.path.join(root, "models", stem, vllm_backend, "config"))
        vllm_candidates.append(os.path.join(root, "models", stem, vllm_backend))
        for path in vllm_candidates:
            normalized = os.path.abspath(path)
            if os.path.isfile(normalized):
                return normalized
            if os.path.isdir(normalized):
                inside_toml = os.path.join(normalized, "default.toml")
                if os.path.isfile(inside_toml):
                    return inside_toml
                inside_json = os.path.join(normalized, "config.json")
                if os.path.isfile(inside_json):
                    return inside_json

    return None


def load_json_config(config_arg: str, backend: str = "hf") -> tuple[dict, str]:
    # Legacy entrypoint name retained for compatibility with existing callers.
    resolved = resolve_config_path(config_arg, backend=backend)
    if not resolved:
        raise FileNotFoundError(
            f"Config file not found for '{config_arg}'. "
            f"Tried direct paths, models/<name>/{backend}/config/default.toml|config.json, "
            f"and legacy config/{backend}/... paths."
        )

    data = _read_config_as_flat(resolved, backend=backend)
    return data, resolved


def load_config_layers(
    config_arg: str,
    *,
    backend: str = "hf",
    profile: str = "",
    include_machine: bool = True,
) -> tuple[dict, dict]:
    base_data, base_path = load_json_config(config_arg, backend=backend)
    layers: list[tuple[str, dict]] = [(base_path, dict(base_data))]
    loaded = [base_path]

    profile_path = ""
    if profile:
        profile_path = _resolve_profile_path(base_path, profile)
        if not os.path.isfile(profile_path):
            raise FileNotFoundError(f"Profile not found: {profile_path}")
        profile_data = _read_config_as_flat(profile_path, backend=backend)
        layers.append((profile_path, dict(profile_data)))
        loaded.append(profile_path)

    machine_path = os.path.join(_repo_root(), "config", "machine.toml")
    if include_machine and os.path.isfile(machine_path):
        pre_merge, _ = _merge_layers_with_origins(layers)
        machine_updates = _machine_override_updates(pre_merge, backend=backend, machine_path=machine_path)
        if machine_updates:
            layers.append((machine_path, machine_updates))
            loaded.append(machine_path)

    merged, origins = _merge_layers_with_origins(layers)
    return merged, {"base": base_path, "profile": profile_path or None, "loaded": loaded, "origins": origins}


def _normalize_model_ref(model_ref: str) -> str:
    raw = os.path.expanduser((model_ref or "").strip())
    win_drive = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
    if win_drive:
        drive = win_drive.group(1).lower()
        rest = win_drive.group(2).replace("\\", "/")
        raw = f"/mnt/{drive}/{rest}"
    return raw


def _model_name_candidates(model_ref: str) -> list[str]:
    value = _normalize_model_ref(model_ref)
    if not value:
        return []

    names: list[str] = []
    gguf_exts = {".gguf"}

    def add_name(raw_name: str) -> None:
        cleaned = (raw_name or "").strip()
        if not cleaned:
            return
        names.append(cleaned)
        stem, ext = os.path.splitext(cleaned)
        if stem and ext.lower() in gguf_exts:
            names.append(stem)

    if value.startswith("ollama:"):
        suffix = value.split(":", 1)[1]
        add_name(suffix)
        add_name(suffix.replace("/", "__").replace(":", "__"))
    elif os.path.exists(value):
        abs_path = os.path.abspath(value)
        parts = abs_path.split(os.sep)
        if "models" in parts:
            idx = parts.index("models")
            if idx + 1 < len(parts):
                add_name(parts[idx + 1])
        add_name(os.path.basename(abs_path.rstrip(os.sep)))
    else:
        add_name(os.path.basename(value.rstrip("/\\")))

    if "/" in value and not os.path.exists(value) and not value.startswith("ollama:"):
        add_name(value.rsplit("/", 1)[-1])

    deduped: list[str] = []
    seen = set()
    for item in names:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def load_default_json_config_for_model(
    model_ref: str,
    backend: str = "hf",
    profile: str = "",
) -> tuple[dict, str] | tuple[None, None]:
    for candidate in _model_name_candidates(model_ref):
        resolved = resolve_config_path(candidate, backend=backend)
        if not resolved:
            continue
        data = _read_config_as_flat(resolved, backend=backend)
        if profile:
            profile_path = _resolve_profile_path(resolved, profile)
            if os.path.isfile(profile_path):
                data = _merge_dicts(data, _read_config_as_flat(profile_path, backend=backend))
        machine_path = os.path.join(_repo_root(), "config", "machine.toml")
        if os.path.isfile(machine_path):
            data = _apply_machine_overrides(data, backend=backend, machine_path=machine_path)
        if isinstance(data, dict):
            return data, resolved
    return None, None


def load_default_config_layers_for_model(
    model_ref: str,
    *,
    backend: str = "hf",
    profile: str = "",
    include_machine: bool = True,
) -> tuple[dict, dict] | tuple[None, None]:
    for candidate in _model_name_candidates(model_ref):
        resolved = resolve_config_path(candidate, backend=backend)
        if not resolved:
            continue

        layers: list[tuple[str, dict]] = [(resolved, _read_config_as_flat(resolved, backend=backend))]
        loaded = [resolved]
        profile_path = ""
        if profile:
            profile_path = _resolve_profile_path(resolved, profile)
            if os.path.isfile(profile_path):
                layers.append((profile_path, _read_config_as_flat(profile_path, backend=backend)))
                loaded.append(profile_path)

        machine_path = os.path.join(_repo_root(), "config", "machine.toml")
        if include_machine and os.path.isfile(machine_path):
            pre_merge, _ = _merge_layers_with_origins(layers)
            machine_updates = _machine_override_updates(pre_merge, backend=backend, machine_path=machine_path)
            if machine_updates:
                layers.append((machine_path, machine_updates))
                loaded.append(machine_path)

        merged, origins = _merge_layers_with_origins(layers)
        return merged, {"base": resolved, "profile": profile_path or None, "loaded": loaded, "origins": origins}
    return None, None
