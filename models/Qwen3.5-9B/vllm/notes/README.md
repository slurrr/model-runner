# Qwen3.5-9B (vLLM) — performance playground notes

This file is intentionally *vLLM-specific*. Model card + HF-specific details live in:
- `models/Qwen3.5-9B/hf/notes/README.md`

## Why vLLM (vs HF/Transformers loop)
vLLM is a serving-oriented engine. The main things you can learn/tune here that aren’t “just Transformers knobs”:
- Scheduler + batching: `--max-num-seqs`, `--max-num-batched-tokens`, `--stream-interval`
- KV cache control: `--gpu-memory-utilization`, `--kv-cache-memory-bytes`, `--block-size`, `--kv-cache-dtype`
- Prefix caching: `--enable-prefix-caching`
- CUDA graphs / compilation: enabled when **not** passing `--enforce-eager`

## Profiles
These configs are intended as tight starting points for experiments:

### Fast throughput (“see how fast it can go”)
- Profile: `fast`
- Run:
  - `python tui.py --backend vllm --config Qwen3.5-9B --profile fast`

What it does:
- Keeps `max_model_len` modest (`8192`) to reduce KV cache pressure.
- Pushes throughput knobs (`--stream-interval`, `--max-num-batched-tokens`, CUDA graph capture size, no request/stats/access logging).
- Enables prefix caching.
- Avoids `--enforce-eager` to keep CUDA-graph/compile paths available.

### Large context (“see how far before OOM”)
- Profile: `lg_ctx`
- Run:
  - `python tui.py --backend vllm --config Qwen3.5-9B --profile lg_ctx`

What it does:
- Increases `max_model_len` (`32768`) and KV reservation (via `gpu_memory_utilization`).
- Enables chunked prefill to make long prompts less painful.
- Keeps `swap-space=0` so “fits or fails” on VRAM only.

## Knobs (what they do + what to try)

### Streaming overhead vs “smoothness”
- `--stream-interval` (default `1`)
  - Bigger = fewer SSE frames = lower CPU/host overhead and often higher throughput, but “bursty” streaming.
  - Try: `1` (smooth) → `4` (good) → `8` (fastest-feeling but chunky).

### Scheduler / batching (even with `max-num-seqs=1` these matter)
- `--max-num-seqs`
  - Max concurrent requests the server will schedule.
  - For single-user TUI benchmarking: keep at `1`.
- `--max-num-batched-tokens`
  - Upper bound for tokens processed per engine iteration (prefill + decode).
  - Bigger can improve throughput, but increases burstiness and can increase peak memory.
  - Try: `4096`, `8192`, `16384` (if you’re not OOM’ing during long prefill).

### KV cache: the “context budget” levers
- `backend.vllm.max_model_len`
  - Hard cap for prompt+generation tokens *that vLLM will accept*.
  - This is your “how far can I push context” dial.
- `--gpu-memory-utilization`
  - Fraction of VRAM vLLM is allowed to use.
  - Bigger usually = more KV cache = higher sustainable context, but too high can make startup fragile.
  - Try: `0.90` → `0.94` → `0.96` → `0.98` (watch for allocator/cudagraph OOM).
- `--kv-cache-memory-bytes`
  - Directly sets KV cache size; **overrides** `--gpu-memory-utilization`.
  - Use when you want deterministic KV sizing while you vary other stuff.
  - Example: `--kv-cache-memory-bytes 16G`
- `--kv-cache-dtype`
  - `auto` uses model dtype. You asked *not* to quantize KV for this model, so keep `auto` (or `bfloat16`).
  - FP8 KV (`fp8*`) can unlock more context, but changes quality/behavior and is its own experiment.
- `--block-size {8,16,32}`
  - KV cache granularity. Smaller may reduce fragmentation but can increase overhead.
  - Try: `32` for throughput, `16` if you’re chasing tight-fit contexts.

### Long-context latency helpers
- `--enable-chunked-prefill`
  - Big win for “huge prompt” responsiveness: it slices prefill work into chunks bounded by `--max-num-batched-tokens`.
- `--long-prefill-token-threshold`
  - Only treat prompts longer than this as “long” for chunked prefill behavior.
  - Try: `1024` or `2048`.

### CUDA graphs / compilation
- `--enforce-eager`
  - Passing this disables cudagraph/compile paths (more stable, often slower).
  - For “go fast”, do **not** pass it.
- `--max-cudagraph-capture-size`
  - Caps which batch sizes get captured; lower reduces capture time/VRAM overhead.
  - Try: `256` (fast startup) vs `512` (potentially more benefit).

### Logging noise (benchmarking vs debugging)
- `--enable-log-requests` / `--no-enable-log-requests`
- `--disable-log-stats`
- `--disable-uvicorn-access-log`
  - Turn these off for perf experiments; turn on selectively when debugging.

## “How do I push it?” (suggested experiment loops)
- Throughput sweep:
  1) keep `max_model_len=8192`
  2) try `stream_interval=1 → 4 → 8`
  3) try `max_num_batched_tokens=4096 → 8192 → 16384`
  4) toggle `enable_prefix_caching` on/off
- Context sweep:
  1) raise `max_model_len` in steps: `16384 → 32768 → 49152 → 65536`
  2) raise `gpu_memory_utilization` gradually (`0.92 → 0.96 → 0.98`)
  3) keep `swap-space=0` so failures are obvious

## Gotchas
- Only set `gen.truncate_prompt_tokens` if you intentionally want vLLM to truncate your prompt; an overly-aggressive value can lead to
  “The decoder prompt cannot be empty”.
- This repo uses a sentinel: `gen.top_k = -1` means “don’t send `top_k` in the vLLM payload”.
- When benchmarking, disable noisy request logging (`--enable-log-requests`) and stats logs unless you’re actively debugging.

## Results log (fill this in as you test)
- `fast`:
  - tokens/s:
  - notes:
- `lg_ctx`:
  - max prompt tokens before failure:
  - tokens/s at 16k/32k:
  - notes:
