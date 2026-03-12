# 0015 - Token Accounting And Metrics

Date: 2026-03-08

## Context
This repo supports multiple backends (HF, GGUF/llama.cpp, EXL2, Ollama, OpenAI-compatible servers, vLLM managed server).

We need consistent and reliable performance/behavior metrics in the TUI:
- prompt token counts
- completion token counts
- total tokens
- tokens/sec

Today, some backends can report exact tokens (token IDs or backend usage), while others only have streamed text.

## Decision
Token counts should come from the most reliable source available, in this order:

1. Backend-native usage counters (preferred)
- Server-provided `usage` fields (OpenAI-compatible servers/vLLM when enabled)
- Engine-provided token IDs (EXL2)

2. Backend tokenization API
- llama.cpp tokenization (`llama_cpp.Llama.tokenize`) for GGUF

3. Retokenization fallback (last resort)
- Only when a tokenizer is already loaded as part of normal operation
- Avoid loading heavyweight tokenizers solely for metrics

UI/display rules:
- `/show` must prefer token metrics when available.
- If token metrics are unavailable, show “unavailable” explicitly (don’t substitute characters and imply they are tokens).
- Character counts may still be shown as a secondary diagnostic signal.

## Consequences
- Adding a new backend requires deciding how token counts will be obtained and documented.
- OpenAI-compatible transports should parse and store usage when the server can provide it.
- Backends that cannot provide reliable token counts should clearly report that limitation in notes and `/show`.

