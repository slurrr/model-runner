#!/usr/bin/env python3
"""Offline FP8 compression workflow for vLLM-compatible artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MethodType
from typing import Any

DEFAULT_OUTPUT_ROOT = "~/models/local/quants"


class ConfigError(Exception):
    """Raised when the compressor config is invalid."""


@dataclass
class CompressionPlan:
    model_name: str
    source_model: str
    tokenizer: str | None
    trust_remote_code: bool
    recipe_scheme: str
    targets: str
    ignore: list[str]
    compressed_name: str
    artifact_dir: Path
    current_link: Path
    calibration_source: str
    dataset_path: str | None
    dataset_id: str | None
    dataset_config: str | None
    dataset_split: str | None
    text_column: str | None
    num_calibration_samples: int
    max_seq_length: int
    shuffle: bool
    batch_size: int
    preprocessing_num_workers: int | None
    dataloader_num_workers: int
    sequential_offload_device: str
    sequential_prefetch: bool
    max_shard_size: str


def import_transformers_runtime():
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

    return AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress a model offline into a vLLM-compatible FP8 artifact."
    )
    parser.add_argument("--model", required=True, help="Repo model name, e.g. OmniCoder-9B.")
    parser.add_argument(
        "--config",
        help="Path to compressor TOML. Defaults to models/<model>/vllm/compressor/default.toml.",
    )
    parser.add_argument(
        "--scheme",
        choices=["fp8"],
        default="fp8",
        help="Compression scheme family. v1 only supports fp8.",
    )
    parser.add_argument("--source-model", help="Override source model path or HF model id.")
    parser.add_argument("--tokenizer", help="Override tokenizer path or HF tokenizer id.")
    parser.add_argument("--output-root", help=f"Output root. Defaults to {DEFAULT_OUTPUT_ROOT}.")
    parser.add_argument("--compressed-name", help="Override the saved compressed model name.")
    parser.add_argument("--run-label", help="Optional run label appended to the timestamp.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code when loading the source model/tokenizer.",
    )
    parser.add_argument(
        "--dataset-source",
        choices=["local", "hf", "none"],
        help="Override calibration source type.",
    )
    parser.add_argument("--dataset-path", help="Override local calibration dataset path.")
    parser.add_argument("--dataset-id", help="Override Hugging Face dataset id.")
    parser.add_argument("--dataset-config", help="Override Hugging Face dataset config name.")
    parser.add_argument("--dataset-split", help="Override calibration dataset split.")
    parser.add_argument("--text-column", help="Override text column for calibration data.")
    parser.add_argument("--num-samples", type=int, help="Override number of calibration samples.")
    parser.add_argument("--max-seq-len", type=int, help="Override max sequence length for calibration.")
    parser.add_argument(
        "--recipe-scheme",
        help="Override low-level FP8 recipe scheme, e.g. FP8, FP8_DYNAMIC, or FP8_BLOCK.",
    )
    parser.add_argument(
        "--targets",
        help="Override quantization targets. Defaults to config value, usually Linear.",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=None,
        help="Append ignore pattern(s) for the quantization recipe. Repeatable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved compression plan and exit.",
    )
    return parser.parse_args()


def default_config_path(model_name: str) -> Path:
    return Path("models") / model_name / "vllm" / "compressor" / "default.toml"


def load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"Compressor config not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc


def require_str(section: dict[str, Any], key: str, context: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Missing required string '{key}' in [{context}]")
    return value.strip()


def require_int(section: dict[str, Any], key: str, context: str) -> int:
    value = section.get(key)
    if not isinstance(value, int):
        raise ConfigError(f"Missing required integer '{key}' in [{context}]")
    return value


def optional_str(section: dict[str, Any], key: str) -> str | None:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"Expected '{key}' to be a string")
    stripped = value.strip()
    return stripped or None


def list_of_str(section: dict[str, Any], key: str, default: list[str] | None = None) -> list[str]:
    value = section.get(key, default or [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"Expected '{key}' to be an array of strings")
    return [item.strip() for item in value if item.strip()]


def resolve_value_path(value: str | None, config_dir: Path) -> str | None:
    if not value:
        return None
    expanded = Path(os.path.expanduser(value))
    if expanded.is_absolute():
        return str(expanded)
    config_relative = (config_dir / expanded).resolve()
    if config_relative.exists():
        return str(config_relative)
    return str(expanded.resolve())


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def scheme_requires_dataset(recipe_scheme: str) -> bool:
    scheme = recipe_scheme.strip().upper()
    return scheme not in {"FP8_DYNAMIC", "FP8_BLOCK"}


def slugify(value: str) -> str:
    pieces: list[str] = []
    for char in value.strip():
        if char.isalnum():
            pieces.append(char.lower())
        elif char in {"-", "_"}:
            pieces.append(char)
    slug = "".join(pieces).strip("-_")
    if not slug:
        raise ConfigError("Run label must contain at least one alphanumeric character")
    return slug


def build_plan(args: argparse.Namespace) -> tuple[CompressionPlan, Path]:
    config_path = Path(args.config) if args.config else default_config_path(args.model)
    config_path = config_path.resolve()
    config = load_toml(config_path)
    config_dir = config_path.parent

    model_cfg = config.get("model", {})
    output_cfg = config.get("output", {})
    calibration_cfg = config.get("calibration", {})
    recipe_cfg = config.get("recipe", {})
    runtime_cfg = config.get("runtime", {})

    if not isinstance(model_cfg, dict) or not isinstance(output_cfg, dict):
        raise ConfigError("Config must define [model] and [output] sections")

    source_model = args.source_model or require_str(model_cfg, "source", "model")
    tokenizer = args.tokenizer or optional_str(model_cfg, "tokenizer")
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False) or args.trust_remote_code)

    compressed_name = args.compressed_name or optional_str(output_cfg, "compressed_name") or f"{args.model}-FP8"
    output_root = Path(
        os.path.expanduser(args.output_root or optional_str(output_cfg, "root") or DEFAULT_OUTPUT_ROOT)
    ).resolve()
    quant_dir_name = optional_str(output_cfg, "quant_dir_name") or "fp8"

    recipe_scheme = args.recipe_scheme or optional_str(recipe_cfg, "scheme") or "FP8"
    targets = args.targets or optional_str(recipe_cfg, "targets") or "Linear"
    ignore = list_of_str(recipe_cfg, "ignore", ["lm_head", "re:.*mlp.gate$"])
    if args.ignore:
        ignore.extend(args.ignore)

    calibration_source = args.dataset_source or (optional_str(calibration_cfg, "source") or "none")
    dataset_path = args.dataset_path or resolve_value_path(optional_str(calibration_cfg, "dataset_path"), config_dir)
    dataset_id = args.dataset_id or optional_str(calibration_cfg, "dataset_id")
    dataset_config = args.dataset_config or optional_str(calibration_cfg, "dataset_config")
    dataset_split = args.dataset_split or optional_str(calibration_cfg, "split")
    text_column = args.text_column or optional_str(calibration_cfg, "text_column")
    num_samples = args.num_samples or require_int(calibration_cfg, "num_calibration_samples", "calibration")
    max_seq_length = args.max_seq_len or require_int(calibration_cfg, "max_seq_length", "calibration")
    shuffle = bool(calibration_cfg.get("shuffle", True))
    batch_size = calibration_cfg.get("batch_size", 1)
    preprocessing_num_workers = calibration_cfg.get("preprocessing_num_workers")
    dataloader_num_workers = calibration_cfg.get("dataloader_num_workers", 0)
    sequential_offload_device = optional_str(runtime_cfg, "sequential_offload_device") or "cuda:0"
    sequential_prefetch = bool(runtime_cfg.get("sequential_prefetch", False))
    max_shard_size = optional_str(output_cfg, "max_shard_size") or "2GB"

    requires_dataset = scheme_requires_dataset(recipe_scheme)
    if calibration_source not in {"local", "hf", "none"}:
        raise ConfigError("calibration.source must be 'local', 'hf', or 'none'")
    if requires_dataset and calibration_source == "none":
        raise ConfigError(
            f"recipe scheme {recipe_scheme!r} requires calibration data; set calibration.source to 'local' or 'hf'"
        )
    if calibration_source == "local":
        if not dataset_path:
            raise ConfigError("Local calibration requires calibration.dataset_path or --dataset-path")
        if not dataset_split:
            raise ConfigError("Local calibration requires calibration.split or --dataset-split")
    if calibration_source == "hf":
        if not dataset_id:
            raise ConfigError("HF calibration requires calibration.dataset_id or --dataset-id")
        if not dataset_split:
            raise ConfigError("HF calibration requires calibration.split or --dataset-split")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ConfigError("calibration.batch_size must be an integer >= 1")
    if preprocessing_num_workers is not None and (
        not isinstance(preprocessing_num_workers, int) or preprocessing_num_workers < 0
    ):
        raise ConfigError("calibration.preprocessing_num_workers must be an integer >= 0")
    if not isinstance(dataloader_num_workers, int) or dataloader_num_workers < 0:
        raise ConfigError("calibration.dataloader_num_workers must be an integer >= 0")

    run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_suffix = slugify(args.run_label) if args.run_label else None
    run_name = f"{run_stamp}-{run_suffix}" if run_suffix else run_stamp

    base_dir = output_root / args.model / quant_dir_name
    artifact_dir = base_dir / "runs" / run_name / compressed_name
    current_link = base_dir / "current"

    plan = CompressionPlan(
        model_name=args.model,
        source_model=source_model,
        tokenizer=tokenizer,
        trust_remote_code=trust_remote_code,
        recipe_scheme=recipe_scheme,
        targets=targets,
        ignore=dedupe(ignore),
        compressed_name=compressed_name,
        artifact_dir=artifact_dir,
        current_link=current_link,
        calibration_source=calibration_source,
        dataset_path=dataset_path,
        dataset_id=dataset_id,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        text_column=text_column,
        num_calibration_samples=num_samples,
        max_seq_length=max_seq_length,
        shuffle=shuffle,
        batch_size=batch_size,
        preprocessing_num_workers=preprocessing_num_workers,
        dataloader_num_workers=dataloader_num_workers,
        sequential_offload_device=sequential_offload_device,
        sequential_prefetch=sequential_prefetch,
        max_shard_size=max_shard_size,
    )
    return plan, config_path


def plan_as_json(plan: CompressionPlan) -> str:
    payload = {
        "model_name": plan.model_name,
        "source_model": plan.source_model,
        "tokenizer": plan.tokenizer,
        "trust_remote_code": plan.trust_remote_code,
        "fp8_recipe": {
            "scheme": plan.recipe_scheme,
            "targets": plan.targets,
            "ignore": plan.ignore,
        },
        "artifact_dir": str(plan.artifact_dir),
        "current_link": str(plan.current_link),
        "calibration": {
            "source": plan.calibration_source,
            "dataset_path": plan.dataset_path,
            "dataset_id": plan.dataset_id,
            "dataset_config": plan.dataset_config,
            "split": plan.dataset_split,
            "text_column": plan.text_column,
            "num_calibration_samples": plan.num_calibration_samples,
            "max_seq_length": plan.max_seq_length,
            "shuffle": plan.shuffle,
            "batch_size": plan.batch_size,
            "preprocessing_num_workers": plan.preprocessing_num_workers,
            "dataloader_num_workers": plan.dataloader_num_workers,
        },
        "runtime": {
            "sequential_offload_device": plan.sequential_offload_device,
            "sequential_prefetch": plan.sequential_prefetch,
        },
        "save": {
            "max_shard_size": plan.max_shard_size,
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def ensure_output_dirs(plan: CompressionPlan) -> None:
    plan.artifact_dir.parent.mkdir(parents=True, exist_ok=False)


def update_current_link(plan: CompressionPlan) -> None:
    link = plan.current_link
    if link.exists() or link.is_symlink():
        if link.is_symlink() or link.is_file():
            link.unlink()
        else:
            raise ConfigError(
                f"Refusing to replace non-symlink directory at {link}. Remove it manually once, then rerun."
            )
    link.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(plan.artifact_dir, link.parent)
    link.symlink_to(relative_target, target_is_directory=True)


def build_recipe(plan: CompressionPlan) -> Any:
    from llmcompressor.modifiers.quantization import QuantizationModifier

    return QuantizationModifier(
        targets=plan.targets,
        scheme=plan.recipe_scheme,
        ignore=plan.ignore,
    )


def load_model_and_processor(plan: CompressionPlan) -> tuple[Any, Any]:
    AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor = import_transformers_runtime()

    model_kwargs = {
        "trust_remote_code": plan.trust_remote_code,
        "dtype": "auto",
    }
    processor_src = plan.tokenizer or plan.source_model

    processor = AutoProcessor.from_pretrained(
        processor_src,
        trust_remote_code=plan.trust_remote_code,
    )

    model = None
    load_errors: list[str] = []
    for loader in (AutoModelForImageTextToText, AutoModelForCausalLM):
        try:
            model = loader.from_pretrained(plan.source_model, **model_kwargs)
            break
        except Exception as exc:
            load_errors.append(f"{loader.__name__}: {exc!r}")

    if model is None:
        joined = "\n".join(load_errors)
        raise ConfigError(f"Failed to load source model with Transformers auto classes:\n{joined}")

    if not hasattr(model, "_get_no_split_modules") and hasattr(model, "_no_split_modules"):
        def _compat_get_no_split_modules(self: Any, device_map: str = "auto") -> list[str]:
            modules = getattr(self, "_no_split_modules", None)
            if not modules:
                return []
            return list(modules)

        model._get_no_split_modules = MethodType(_compat_get_no_split_modules, model)

    return model, processor


def configure_save_path(model: Any, max_shard_size: str) -> None:
    original = model.save_pretrained

    def _save_with_shard_size(self: Any, save_directory: str, *args: Any, **kwargs: Any):
        kwargs.setdefault("max_shard_size", max_shard_size)
        return original(save_directory, *args, **kwargs)

    model.save_pretrained = MethodType(_save_with_shard_size, model)


def write_manifest(plan: CompressionPlan, config_path: Path) -> None:
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "model_name": plan.model_name,
        "source_model": plan.source_model,
        "tokenizer": plan.tokenizer,
        "compressed_name": plan.compressed_name,
        "scheme_family": "fp8",
        "recipe_scheme": plan.recipe_scheme,
        "targets": plan.targets,
        "ignore": plan.ignore,
        "artifact_dir": str(plan.artifact_dir),
        "current_link": str(plan.current_link),
        "config_path": str(config_path),
        "calibration": {
            "source": plan.calibration_source,
            "dataset_path": plan.dataset_path,
            "dataset_id": plan.dataset_id,
            "dataset_config": plan.dataset_config,
            "split": plan.dataset_split,
            "text_column": plan.text_column,
            "num_calibration_samples": plan.num_calibration_samples,
            "max_seq_length": plan.max_seq_length,
            "shuffle": plan.shuffle,
            "batch_size": plan.batch_size,
            "preprocessing_num_workers": plan.preprocessing_num_workers,
            "dataloader_num_workers": plan.dataloader_num_workers,
        },
        "runtime": {
            "sequential_offload_device": plan.sequential_offload_device,
            "sequential_prefetch": plan.sequential_prefetch,
        },
        "save": {
            "max_shard_size": plan.max_shard_size,
        },
    }
    manifest_path = plan.artifact_dir / "compression-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def validate_runtime_inputs(plan: CompressionPlan) -> None:
    if plan.calibration_source == "local":
        dataset_file = Path(plan.dataset_path or "")
        if not dataset_file.exists():
            raise ConfigError(f"Local calibration dataset does not exist: {dataset_file}")
        if dataset_file.name.lower() == "readme.md":
            raise ConfigError("Replace the calibration README placeholder with a real dataset path before compressing")


def run_compression(plan: CompressionPlan, config_path: Path) -> None:
    try:
        from llmcompressor import oneshot
    except ImportError:
        try:
            from llmcompressor.entrypoints.oneshot import oneshot
        except ImportError as exc:
            raise SystemExit(
                "llmcompressor is not installed in this environment. Install a compatible llmcompressor stack before running compression."
            ) from exc

    validate_runtime_inputs(plan)
    recipe = build_recipe(plan)
    ensure_output_dirs(plan)
    model, processor = load_model_and_processor(plan)
    configure_save_path(model, plan.max_shard_size)

    oneshot_kwargs: dict[str, Any] = {
        "model": model,
        "processor": processor,
        "recipe": recipe,
        "output_dir": str(plan.artifact_dir),
    }
    stage_dir: Path | None = None
    if plan.calibration_source != "none":
        oneshot_kwargs.update(
            {
                "num_calibration_samples": plan.num_calibration_samples,
                "max_seq_length": plan.max_seq_length,
                "splits": plan.dataset_split,
                "shuffle_calibration_samples": plan.shuffle,
                "pipeline": "sequential",
                "batch_size": plan.batch_size,
                "preprocessing_num_workers": plan.preprocessing_num_workers,
                "dataloader_num_workers": plan.dataloader_num_workers,
                "sequential_offload_device": plan.sequential_offload_device,
                "sequential_prefetch": plan.sequential_prefetch,
            }
        )
        if plan.text_column:
            oneshot_kwargs["text_column"] = plan.text_column
    if plan.calibration_source == "local":
        dataset_file = Path(plan.dataset_path or "")
        stage_dir = Path(tempfile.mkdtemp(prefix="compress-calibration-"))
        staged_dataset = stage_dir / "train.json"
        shutil.copy2(dataset_file, staged_dataset)
        oneshot_kwargs["dataset"] = "json"
        oneshot_kwargs["dataset_path"] = str(stage_dir)
    elif plan.calibration_source == "hf":
        oneshot_kwargs["dataset"] = plan.dataset_id
        if plan.dataset_config:
            oneshot_kwargs["dataset_config_name"] = plan.dataset_config

    try:
        oneshot(**oneshot_kwargs)
        write_manifest(plan, config_path)
        update_current_link(plan)
    finally:
        if stage_dir is not None:
            shutil.rmtree(stage_dir, ignore_errors=True)


def main() -> int:
    args = parse_args()
    try:
        plan, config_path = build_plan(args)
        if args.dry_run:
            print(plan_as_json(plan))
            return 0
        run_compression(plan, config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
