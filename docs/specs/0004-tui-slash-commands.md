# Spec: TUI slash commands (`/help`, `/show`, `/system`, …)

Date: 2026-03-02

## Goal
Add a robust, extensible slash-command system to the Textual TUI so you can inspect (and later tweak) anything you might want to verify while chatting.

Initial requirements:
- `/system` shows the current system prompt (and its source).
- `/prefix` shows user/prompt prefixes (when applicable).
- `/show` displays current session parameters / state.
- `/?` shows all available slash commands.
- Commands with sub-arguments should be self-documenting:
  - typing `/show` with no args prints available `show` topics.

## Non-goals (v1)
- In-place editing of config files.
- Arbitrary runtime mutation of backend settings while a generation is running.
- Tool execution / agent UI.

## Definitions
- **Slash command**: a line submitted in the TUI input that starts with `/` (e.g. `/show gen`).
- **Command output**: local-only information rendered into the transcript (not sent to the model and not appended to the chat history).

## UX

### Baseline behavior
- If user input starts with `/`, parse it as a slash command.
- If command is recognized:
  - run it locally
  - render output as an “info” message in the transcript (distinct styling)
  - do not send it to the model
  - do not append it to `self.messages` history
- If command is not recognized:
  - show a short error plus closest matches (fuzzy suggestions)

### Help and discoverability
- `/?` and `/help` are synonyms.
- `/help` lists commands (name + 1-line summary).
- `/help <cmd>` shows detailed help (aliases, usage, examples).
- `/show` with no args prints:
  - usage: `/show <topic>`
  - list of available topics
  - example invocations

### Output placement
Command output should be appended to the transcript so it is scrollable and captured in the session flow.

Suggested styling:
- Prefix command echoes with something like: `# /show gen`
- Output in dim/grey or yellow headers, keeping it readable but clearly “local”.

### Generation-in-progress behavior
Slash commands are allowed while a generation is running, but must be read-only in v1.
- If a command implies mutation (future), show “Not allowed during generation”.

## Commands (v1)

### Core help / discovery
- `/?` or `/help`:
  - list all commands
- `/help <cmd>`:
  - detailed help for `cmd`

### Primary introspection surface: `/show`
- `/show`:
  - list available show topics (see below)
  - show brief usage examples
- `/show <topic>`:
  - print details for `topic`
- `/show <topic> ?` or `/show <topic> --help`:
  - print available fields/subtopics for that topic (when applicable)

### Prompt inspection (single-purpose shortcuts)
- `/system`:
  - show current system prompt content as used by the app
  - include source hints:
    - whether it came from `--system` or `--system-file`
    - resolved path to the system file (if used)
- `/prefix`:
  - show:
    - `user_prefix` (TUI common)
    - `prompt_prefix` only if the active app/backend supports it (print “N/A” otherwise)
    - `prompt_mode` (chat vs plain)
    - chat template override source (if applicable; HF only)

### Model / config / environment shortcuts
These are convenience aliases for `/show <topic>` (single implementation).

- `/model` == `/show model`
- `/config` == `/show config`
- `/env` == `/show env`
- `/files` == `/show files`

### Session utilities (existing + common)
- `/clear`: clear transcript + conversation history (already exists)
- `/exit` or `/quit`: exit TUI (already exists)

### `/show` topics (v1)
`/show` is the primary “introspection surface”. Topics should be stable and additive.

Suggested topics:
- `/show session`
  - backend name, resolved model id, config path (if any)
  - `is_generating`, `follow_output`
  - transcript stats (num widgets / scroll position summary)
- `/show prompt`
  - system prompt presence, system file, user_prefix, prompt_mode (if relevant)
- `/show gen`
  - max_new_tokens, temperature, top_p, top_k, stop strings
  - backend-specific generation settings (only those that apply)
- `/show ui`
  - show_thinking, animation, scroll_lines, ui_tick_ms, ui_max_events_per_tick
- `/show args`
  - dump the parsed CLI args (filtered for readability)
- `/show history`
  - number of turns, number of messages, last N roles
- `/show last`
  - last `TurnRecord` summary (timing, ended_in_think, lengths of raw/think/answer)
  - (No additional computation in v1; only display fields already captured today.)

Optional topics (if trivial from existing state):
- `/show env` (CUDA available, `OLLAMA_HOST` presence, etc.)
- `/show files` (resolved system file path, save_transcript path)
- `/show model` (backend + resolved model id; same as `/model`)
- `/show config` (loaded config path; same as `/config`)

### Aliases (v1)
These should be implemented as simple dispatch to the canonical handler (to avoid drift).

Help / discovery:
- `/?` == `/help`

`/show` topic aliases:
- `/model` == `/show model`
- `/config` == `/show config`
- `/env` == `/show env`
- `/files` == `/show files`

Optional “shortcuts” (recommended if they don’t create ambiguity):
- `/session` == `/show session`
- `/prompt` == `/show prompt`
- `/gen` == `/show gen`
- `/ui` == `/show ui`
- `/args` == `/show args`
- `/history` == `/show history`
- `/last` == `/show last`

## Implementation design

### Parsing
- Treat a submitted line beginning with `/` as a slash command.
- Parse tokens using `shlex.split` so quoting works for future commands.
  - Example: `/show args --json`
- Command name is the first token (without leading `/`).
- Remaining tokens are `argv` for that command.

### Registry
Implement a small command registry for discoverability and “`/show` with no args prints options”.

Data model (conceptual):
- `SlashCommand`:
  - `name` (primary)
  - `aliases`
  - `summary`
  - `usage`
  - `handler(app, argv) -> str | Text | list[Text]`
- `SlashRegistry`:
  - `register(cmd)`
  - `resolve(name_or_alias) -> cmd`
  - `help_all()`
  - `help_one(cmd)`

### Command output widget
Add a lightweight transcript widget for local-only output (e.g. `InfoMessage`):
- Accepts a Rich `Text` (or string)
- Styles as dim/grey; uses wrapping (`width: 100%`, `text-wrap: wrap`)

### Keep “comprehensive” without adding computation
This feature should prioritize **visibility of already-known values** over new instrumentation.

Concretely:
- `/show gen` should print the effective values that will be sent to the backend (from parsed args/config).
- `/show last` should print whatever is already captured in `TurnRecord` (e.g. `timing.elapsed`) but should not introduce new token counting / profiling logic in v1.

### Backend-specific data (optional, minimal)
For `/show backend` (or as part of `/show session`), the app can optionally query backend sessions for extra info without changing the `BackendSession` protocol:
- If session has `get_info()` / `describe()` (duck-typed), call it and include key/value pairs.
- Otherwise display only:
  - `session.backend_name`
  - `session.resolved_model_id`

## Edge cases
- Unknown command: show closest matches (case-insensitive).
- Empty `/` input: treat as `/help`.
- Commands should not be stored in the model conversation history.
- Large outputs (e.g. `/show args`):
  - keep output concise by default (top-level keys only)
  - support `--all` or `--json` later if needed

## Acceptance checklist
- `/?` lists commands.
- `/show` lists show topics (and examples).
- `/system` prints the active system prompt and source.
- `/prefix` prints prefixes (and “N/A” when not applicable).
- `/model`, `/config`, `/env`, `/files` work and do not pollute history.
- Slash commands do not get sent to the model and do not pollute history.
