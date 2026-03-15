# Model Runner -> OpenClaw Compatibility Checklist

## Must support

- `GET /v1/models`
- `POST /v1/chat/completions`
- SSE streaming
- request `tools` array
- assistant tool calls
- follow-up `tool` role messages

## Message roles

Your backend must accept:

- `system`
- `user`
- `assistant`
- `tool`

Note:

- OpenClaw forces non-native OpenAI-compatible backends to avoid `developer` role in `src/agents/model-compat.ts:39`

## Tool call streaming

Stream tool calls in OpenAI-compatible delta form:

- `choices[0].delta.tool_calls[*].index`
- `choices[0].delta.tool_calls[*].id`
- `choices[0].delta.tool_calls[*].type`
- `choices[0].delta.tool_calls[*].function.name`
- `choices[0].delta.tool_calls[*].function.arguments`

Arguments may arrive in fragments.
OpenClaw/tool-capable clients expect incremental accumulation.

## Tool loop behavior

Your backend must correctly handle:

1. assistant emits tool call(s)
2. client executes tool(s)
3. next request includes:
   - assistant message with `tool_calls`
   - tool message(s) with `tool_call_id`
4. model continues from tool result

## Request compatibility expectations

Safest assumptions:

- accept `parallel_tool_calls: false`
- accept `tool_choice` if sent
- accept `max_tokens`
- ignore unknown safe fields instead of crashing
- do not require OpenAI-only hosted features

## Recommended local defaults

- low temperature (`0.0` to `0.2`)
- `parallel_tool_calls: false`
- one tool call at a time
- prefer accurate function-name emission over creativity

## Success criteria

A run is good enough for POC when:

- the model picks the right tool name
- the JSON args parse cleanly
- tool call ids are stable
- the model correctly consumes tool results
- it can finish a small edit/test/fix loop without hallucinating unavailable tools
