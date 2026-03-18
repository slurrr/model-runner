# Model Runner -> OpenClaw Compatibility Checklist

## Must support

- `GET /v1/models`
- `POST /v1/chat/completions`
- SSE streaming
- request `tools` array
- assistant tool calls
- follow-up `tool` role messages
- `stream=false` chat completions for compatibility checks

## Message roles

Your backend must accept:

- `system`
- `user`
- `assistant`
- `tool`

Notes:
- OpenClaw avoids `developer` role for non-native OpenAI-compatible backends.
- `role="tool"` messages may include an optional `name` field; tolerate it even if it is not semantically required.

## Tool call streaming

Stream tool calls in OpenAI-compatible delta form:

- `choices[0].delta.tool_calls[*].index`
- `choices[0].delta.tool_calls[*].id`
- `choices[0].delta.tool_calls[*].type`
- `choices[0].delta.tool_calls[*].function.name`
- `choices[0].delta.tool_calls[*].function.arguments`

Arguments may arrive in fragments.
Clients must accumulate them incrementally.

Handle these cases correctly:
- assistant `content` is `null`, empty, or whitespace while tool calls stream
- finish arrives as `finish_reason="tool_calls"`

## Tool loop behavior

Your backend must correctly handle:

1. assistant emits tool call(s)
2. client executes tool(s)
3. next request includes:
   - assistant message with `tool_calls`
   - tool message(s) with `tool_call_id`
4. model continues from tool result

Assistant tool-call messages and their tool results should remain protocol-consistent across follow-up turns.

## Request compatibility expectations

Safest assumptions:

- accept `parallel_tool_calls: false`
- accept `tool_choice` when sent
- accept `max_tokens`
- ignore unknown safe fields instead of crashing
- do not require hosted-only OpenAI features

Accepted `tool_choice` shapes:
- `"auto"`
- `"none"`
- object form selecting a named function

## SSE behavior

Streaming implementations should tolerate:
- `event:` fields
- unknown SSE fields
- multi-line `data:` blocks
- `[DONE]` end marker

Server-side errors should surface clearly when received as:
- top-level `{"error": {...}}`
- an `event: error` frame
- a non-standard error envelope carrying readable error text

## Non-stream response behavior

For `stream=false`, check:
- `choices[0].message.content`
- `choices[0].message.tool_calls`
- `usage` when present

## Multimodal stance

Current stance for model-runner OpenAI-compatible requests:
- OpenAI-style content parts arrays are supported
- TUI image attachments are converted to `image_url` content parts using local data URLs
- local size limits still apply

For OpenClaw integration, this should be treated as vLLM-first behavior for now.

## Recommended local defaults

- low temperature (`0.0` to `0.2`) for tool-heavy loops
- `parallel_tool_calls: false`
- one tool call at a time in early tuning passes
- prefer accurate function-name emission over creativity

## Success criteria

A run is good enough for POC when:

- the model picks the right tool name
- the JSON args parse cleanly
- tool call ids are stable
- the model correctly consumes tool results
- `finish_reason` is sensible (`tool_calls` or normal stop)
- server-side errors are surfaced clearly instead of silently hanging or looping
- it can finish a small edit/test/fix loop without hallucinating unavailable tools
