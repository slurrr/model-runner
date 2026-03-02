import json
import os
import re


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def resolve_config_path(config_arg: str, backend: str = "hf") -> str | None:
    if not config_arg:
        return None

    raw = os.path.expanduser(config_arg.strip())
    root = _repo_root()

    candidates = []
    candidates.append(raw)
    if not raw.endswith(".json"):
        candidates.append(f"{raw}.json")

    # Relative to repo root
    candidates.append(os.path.join(root, raw))
    if not raw.endswith(".json"):
        candidates.append(os.path.join(root, f"{raw}.json"))

    basename = os.path.basename(raw)
    stem = basename[:-5] if basename.endswith(".json") else basename
    if stem == "template":
        stem = "_TEMPLATE"

    # Model-first layout (preferred), e.g. models/Nanbeige4.1-3B/hf/config/config.json
    candidates.append(os.path.join(root, "models", stem, backend, "config", "config.json"))
    candidates.append(os.path.join(root, "models", stem, backend, "config"))
    candidates.append(os.path.join(root, "models", stem, backend))
    candidates.append(os.path.join(root, "models", stem))

    # Backend-scoped fallbacks (legacy), e.g. config/hf/Nanbeige4.1-3B.json
    candidates.append(os.path.join(root, "config", backend, basename))
    if not basename.endswith(".json"):
        candidates.append(os.path.join(root, "config", backend, f"{basename}.json"))

    # Backend-scoped folder layout (legacy), e.g. config/hf/Nanbeige4.1-3B/config.json
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
            inside = os.path.join(normalized, "config.json")
            if os.path.isfile(inside):
                return inside

    return None


def load_json_config(config_arg: str, backend: str = "hf") -> tuple[dict, str]:
    resolved = resolve_config_path(config_arg, backend=backend)
    if not resolved:
        raise FileNotFoundError(
            f"Config file not found for '{config_arg}'. "
            f"Tried direct paths, models/<name>/{backend}/config/config.json, and legacy config/{backend}/... paths."
        )

    with open(resolved, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object at the top level.")

    return data, resolved


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

    if os.path.exists(value):
        abs_path = os.path.abspath(value)
        parts = abs_path.split(os.sep)
        if "models" in parts:
            idx = parts.index("models")
            if idx + 1 < len(parts):
                names.append(parts[idx + 1])
        names.append(os.path.basename(abs_path.rstrip(os.sep)))
    else:
        names.append(os.path.basename(value.rstrip("/\\")))

    # For HF IDs like org/model, use final segment.
    if "/" in value and not os.path.exists(value):
        names.append(value.rsplit("/", 1)[-1])

    deduped: list[str] = []
    seen = set()
    for item in names:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def load_default_json_config_for_model(model_ref: str, backend: str = "hf") -> tuple[dict, str] | tuple[None, None]:
    for candidate in _model_name_candidates(model_ref):
        resolved = resolve_config_path(candidate, backend=backend)
        if not resolved:
            continue
        with open(resolved, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data, resolved
    return None, None
