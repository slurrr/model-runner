from __future__ import annotations

from collections import OrderedDict

ALL_GEN_KNOBS = [
    "allowed_token_ids",
    "best_of",
    "frequency_penalty",
    "ignore_eos",
    "include_stop_str_in_output",
    "length_penalty",
    "max_new_tokens",
    "max_time",
    "min_p",
    "min_tokens",
    "no_repeat_ngram_size",
    "num_beams",
    "presence_penalty",
    "prompt_logprobs",
    "repetition_penalty",
    "seed",
    "skip_special_tokens",
    "spaces_between_special_tokens",
    "stop_strings",
    "stop_token_ids",
    "temperature",
    "top_k",
    "top_p",
    "truncate_prompt_tokens",
    "typical_p",
    "use_beam_search",
]

SUPPORTED_KNOBS = {
    "hf": {
        "max_new_tokens",
        "max_time",
        "min_p",
        "no_repeat_ngram_size",
        "num_beams",
        "repetition_penalty",
        "stop_strings",
        "temperature",
        "top_k",
        "top_p",
        "typical_p",
    },
    "gguf": {
        "max_new_tokens",
        "min_p",
        "repetition_penalty",
        "stop_strings",
        "temperature",
        "top_k",
        "top_p",
        "typical_p",
    },
    "exl2": {
        "frequency_penalty",
        "max_new_tokens",
        "min_p",
        "presence_penalty",
        "repetition_penalty",
        "stop_strings",
        "temperature",
        "top_k",
        "top_p",
        "typical_p",
    },
    "openai": {
        "max_new_tokens",
        "seed",
        "stop_strings",
        "temperature",
        "top_p",
    },
    "vllm": {
        "allowed_token_ids",
        "best_of",
        "frequency_penalty",
        "ignore_eos",
        "include_stop_str_in_output",
        "length_penalty",
        "max_new_tokens",
        "min_p",
        "min_tokens",
        "presence_penalty",
        "prompt_logprobs",
        "repetition_penalty",
        "seed",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "stop_strings",
        "stop_token_ids",
        "temperature",
        "top_k",
        "top_p",
        "truncate_prompt_tokens",
        "use_beam_search",
    },
    "ollama": {
        "max_new_tokens",
        "stop_strings",
        "temperature",
        "top_k",
        "top_p",
    },
}


def user_set_keys(args) -> set[str]:
    cli_overrides = set(getattr(args, "_cli_overrides", set()) or set())
    config_keys = set(getattr(args, "_config_keys", set()) or set())
    return cli_overrides | config_keys


def unsupported_user_set(args, backend: str) -> list[str]:
    supported = SUPPORTED_KNOBS.get(backend, set())
    keys = user_set_keys(args)
    return sorted(k for k in keys if k in ALL_GEN_KNOBS and k not in supported)


def finalize_knob_report(
    *,
    sent: dict[str, object],
    supported: set[str],
    ignored: list[str] | None = None,
    notes: list[str] | None = None,
    mode: str | None = None,
) -> dict[str, object]:
    ignored_sorted = sorted(set(ignored or []))
    sent_sorted = OrderedDict((key, sent[key]) for key in sorted(sent.keys()))
    deferred_sorted = sorted(key for key in supported if key not in sent_sorted and key not in ignored_sorted)
    out: dict[str, object] = {
        "sent": dict(sent_sorted),
        "deferred": deferred_sorted,
        "ignored": ignored_sorted,
    }
    if mode:
        out["mode"] = mode
    note_rows = [str(note) for note in (notes or []) if str(note).strip()]
    if note_rows:
        out["notes"] = note_rows
    return out


def build_intent_knobs(args, backend: str) -> dict[str, object]:
    sent: dict[str, object] = {}
    if backend == "hf":
        sent = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "num_beams": args.num_beams,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
        }
        if args.top_k is not None:
            sent["top_k"] = args.top_k
        if args.stop_strings is not None:
            sent["stop_strings"] = args.stop_strings
        if args.typical_p is not None:
            sent["typical_p"] = args.typical_p
        if args.min_p is not None:
            sent["min_p"] = args.min_p
        if args.max_time is not None:
            sent["max_time"] = args.max_time
    elif backend == "gguf":
        sent = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        if args.top_k is not None:
            sent["top_k"] = args.top_k
        if args.min_p is not None:
            sent["min_p"] = args.min_p
        if args.typical_p is not None:
            sent["typical_p"] = args.typical_p
        if args.repetition_penalty not in (None, 1.0):
            sent["repetition_penalty"] = args.repetition_penalty
        if args.stop_strings is not None:
            sent["stop_strings"] = args.stop_strings
    elif backend == "exl2":
        sent = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        if args.top_k is not None:
            sent["top_k"] = args.top_k
        if args.min_p is not None:
            sent["min_p"] = args.min_p
        if args.typical_p is not None:
            sent["typical_p"] = args.typical_p
        if args.repetition_penalty is not None:
            sent["repetition_penalty"] = args.repetition_penalty
        if args.frequency_penalty is not None:
            sent["frequency_penalty"] = args.frequency_penalty
        if args.presence_penalty is not None:
            sent["presence_penalty"] = args.presence_penalty
        if args.stop_strings:
            sent["stop_strings"] = args.stop_strings
    elif backend == "openai":
        sent = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        if args.stop_strings:
            sent["stop_strings"] = args.stop_strings
        if args.seed is not None:
            sent["seed"] = args.seed
    elif backend == "vllm":
        sent = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        for key in (
            "top_k",
            "min_p",
            "stop_strings",
            "stop_token_ids",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "seed",
            "truncate_prompt_tokens",
            "allowed_token_ids",
            "prompt_logprobs",
        ):
            value = getattr(args, key, None)
            if value not in (None, [], 0, 0.0):
                sent[key] = value
        if args.ignore_eos is True:
            sent["ignore_eos"] = True
        if args.min_tokens not in (None, 0):
            sent["min_tokens"] = args.min_tokens
        if args.best_of not in (None, 1):
            sent["best_of"] = args.best_of
        if args.use_beam_search is True:
            sent["use_beam_search"] = True
        if args.length_penalty not in (None, 1.0):
            sent["length_penalty"] = args.length_penalty
        if args.include_stop_str_in_output is True:
            sent["include_stop_str_in_output"] = True
        if args.skip_special_tokens is False:
            sent["skip_special_tokens"] = False
        if args.spaces_between_special_tokens is False:
            sent["spaces_between_special_tokens"] = False
    elif backend == "ollama":
        user_set = user_set_keys(args)
        if "temperature" in user_set and args.temperature is not None:
            sent["temperature"] = args.temperature
        if "top_p" in user_set and args.top_p is not None:
            sent["top_p"] = args.top_p
        if "top_k" in user_set and args.top_k is not None:
            sent["top_k"] = args.top_k
        if "max_new_tokens" in user_set and args.max_new_tokens is not None:
            sent["max_new_tokens"] = args.max_new_tokens
        if "stop_strings" in user_set and args.stop_strings is not None:
            sent["stop_strings"] = args.stop_strings

    return finalize_knob_report(
        sent=sent,
        supported=SUPPORTED_KNOBS.get(backend, set()),
        ignored=unsupported_user_set(args, backend),
        mode="intent",
    )
