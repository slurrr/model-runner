#!/usr/bin/env markdown
# Spec: Safe “browser” tool v1 (fetch-only, prompt-injection resistant)

Date: 2026-03-11

Status: Draft

Related specs:
- `docs/specs/0020-tui-mvp-tool-execution-harness.md` (tool loop)
- `docs/specs/0011-openclaw-tool-compat-layer.md` (schema + normalization concepts)
- `docs/specs/0017-tui-logging-ring-buffer.md` (logging contract)

## Context / problem
Once a model has a “browser” tool, risks jump significantly:
- prompt injection from web content
- SSRF / internal network probing
- unbounded crawling / huge downloads
- accidental side effects (forms, actions)

We want a browser tool for local agent testing in this repo **without** opening the full dangerous surface area.

This spec defines a safe v1 browser tool that is:
- GET-only
- no cookies/auth/session state
- strict size/time budgets
- hard blocks private/localhost ranges by default
- returns content wrapped as **untrusted**

## Goals
- Provide a single, simple browser tool usable in the TUI tool harness.
- Make it safe enough to test “tool calling” behavior without turning the repo into a crawler.
- Make debugging easy (logs + transcript blocks show exactly what happened).

## Non-goals (v1)
- Clicking, form submission, POST/PUT/DELETE.
- JS rendering / headless browser automation.
- File downloads beyond small text/HTML.
- Automatic multi-hop browsing (agent loop across many pages). (We can add later behind budgets.)

## Tool surface (v1)
Expose exactly one function tool:

### `browser_fetch`
Signature:
- `url: str` (required)

Optional args (v1):
- `max_bytes: int` (optional; capped by config hard limit)

Returns:
- UTF-8 text (sanitized) wrapped in an “untrusted content” envelope.

Notes:
- The tool MUST only support `http://` and `https://`.
- The tool MUST only perform GET.
- Redirects MAY be followed up to a small limit (default 3) while re-applying policy for each hop.

## Config (TOML)
Extend the tool harness config with a browser subsection:

```toml
[tools.browser]
enabled = false

# Safety policy (defaults are strict)
allow_domains = []            # if non-empty: only these domains (and subdomains) are allowed
deny_domains = []             # deny wins
deny_private_ips = true       # block 127.0.0.1, RFC1918, link-local, etc.
deny_ports = [25, 110, 143]   # optional (example); default empty means “allow standard ports”

# Budgets
timeout_s = 10
max_bytes_per_fetch = 2_000_000
max_fetches_per_turn = 2
max_fetches_per_session = 20
max_redirects = 3

# Output shaping
include_http_metadata = true  # status, final_url, content_type, bytes_read
strip_html = true             # HTML -> readable text
max_output_chars = 8000       # truncate returned text (after stripping)
```

Policy notes:
- If `allow_domains` is empty, treat as “allow-all” **except** for `deny_private_ips` and `deny_domains`.
- For agent safety, recommended default remains: set `allow_domains` for any real browsing sessions.

## Safety invariants (MUST)
Regardless of allowlist settings:
- Reject any non-http(s) URL schemes.
- Reject URLs with embedded credentials (`user:pass@host`).
- Resolve host → IP and block private/local/link-local ranges when `deny_private_ips=true`.
- Enforce strict timeouts and size caps (stream and cut off at `max_bytes_per_fetch`).
- Do not send cookies; do not persist cookies; do not follow `Set-Cookie`.
- Do not include response headers that could contain secrets (default: do not return headers at all).
- Truncate tool output to `max_output_chars` before returning to the model.

## Prompt-injection handling (MUST)
Returned content MUST be wrapped as untrusted, e.g.:

```
UNTRUSTED_WEB_CONTENT_START
source_url: ...
...
<content>
...
</content>
UNTRUSTED_WEB_CONTENT_END
```

Additionally, the tool harness MUST inject a tool-safety policy block into the model’s system prompt when browser tools are enabled:
- explicitly state web content is untrusted
- explicitly forbid following instructions found in web content
- explicitly instruct the model to only extract facts / answer the user

This policy block SHOULD be dynamically generated from config (domains/budgets) so the model sees the real constraints.

## Logging + transcript UX
For each fetch, log (via session logger):
- requested URL
- final URL (after redirects)
- resolved IP(s)
- status code
- bytes read
- elapsed time
- policy decision (allowed/blocked + reason)

Transcript tool block MUST show:
- tool name (`browser_fetch`)
- url + final_url
- status / bytes
- either:
  - returned (truncated) content preview, or
  - error + policy reason

## Error handling (MUST)
If blocked by policy:
- return a tool error result (do not execute request)
- include a short reason (e.g. `blocked_private_ip`, `domain_not_allowlisted`, `scheme_not_allowed`)

If fetch fails (DNS, TLS, timeout, too large):
- return a tool error result with a short classified reason.

## Testing checklist
- Try fetching `http://127.0.0.1` with default config:
  - must be blocked.
- Fetch a small public HTML page under an allowlisted domain:
  - returns stripped readable text, wrapped as untrusted.
- Fetch a large file:
  - truncates by bytes and/or rejects with `too_large`.
- Prompt injection attempt in page content:
  - model should not comply; logs should show wrapper; system policy should be present.

