# gemma-3-27b-it-abliterated-exl2 templates (exl2)

This folder is for prompt/template overrides or experiments.

Suggested starting points:
- `gemma_chat_sanitized.jinja`: Gemma turn tokens + system-prefix injection + strips pre-`</think>` from assistant history
- `gemma_chat_sanitized_assume_think.jinja`: same, but adds a `<think>` prompt prefix (pair with `assume_think=true` + an explicit `</think>` instruction)
