from __future__ import annotations

import argparse
import os

from tui_app.log_file import FileLogger
from tui_app.transports.openai_http import OpenAIHTTPSession, normalize_openai_base_url, resolve_model_once


def _resolve_api_key(args: argparse.Namespace) -> str:
    cli_overrides = set(getattr(args, "_cli_overrides", set()) or set())
    if "openai_api_key" in cli_overrides:
        return (args.openai_api_key or "").strip()
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    return (args.openai_api_key or "").strip()


def create_session(args: argparse.Namespace) -> OpenAIHTTPSession:
    base_url = normalize_openai_base_url((args.openai_base_url or "").strip())
    if not base_url:
        raise RuntimeError("OpenAI backend requires base_url (--openai-base-url or config [backend.openai].base_url).")

    api_key = _resolve_api_key(args)

    logger = FileLogger.from_value(
        getattr(args, "openai_log_file", ""),
        "backend",
        config_path=getattr(args, "_config_path", None),
    )
    resolved_model_id = (args.model_id or "").strip()
    if resolved_model_id.startswith("openai:"):
        resolved_model_id = resolved_model_id.split(":", 1)[1].strip()
    if not resolved_model_id:
        resolved_model_id = resolve_model_once(
            base_url,
            timeout_s=float(args.openai_timeout_s),
            api_key=api_key,
        )
    if logger is not None:
        logger.log(
            f"session_init base_url={base_url} model={resolved_model_id} timeout_s={float(args.openai_timeout_s)}",
            source="app",
        )
        logger.log("backend_ready openai session attached", source="backend")
    requested_template = (args.chat_template or "").strip()

    return OpenAIHTTPSession(
        args=args,
        resolved_model_id=resolved_model_id,
        base_url=base_url,
        api_key=api_key,
        timeout_s=float(args.openai_timeout_s),
        backend_name="openai",
        template_info={
            "template_control_level": "server_owned_template",
            "chat_template_requested": requested_template,
            "chat_template_applied": False,
            "chat_template_reason": "ignored_server_owned" if requested_template else "empty_default",
        },
        logger=logger,
    )
