# lemon

A chatbot that simulates a friend, not an assistant. Runs on OpenRouter (Claude by default), keeps an internal emotional state, remembers facts about you across sessions, and types like a person.

Two frontends share the same backend: a CLI REPL and a single-page web UI.

## Setup

```bash
pip install -r requirements.txt
```

Create `.env` with your OpenRouter key:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

Optional overrides (also via env var):

| variable             | default                          | meaning                                      |
| -------------------- | -------------------------------- | -------------------------------------------- |
| `LEMON_CHAT_MODEL`   | `anthropic/claude-sonnet-4.6`    | main chat model                              |
| `LEMON_STATE_MODEL`  | `anthropic/claude-haiku-4.5`     | small model that nudges internal state       |
| `LEMON_PROMPT_CACHE` | `1`                              | toggle Anthropic-style prompt caching        |
| `LEMON_HUMANIZE`     | `1`                              | toggle per-token typing pacing               |
| `LEMON_DB`           | `.lemon.db`                      | SQLite path (gitignored)                     |

## Run

CLI:
```bash
python src/lem.py
```

Web UI:
```bash
python src/web.py
# then visit http://127.0.0.1:8000
```

## Slash commands

Type `/help` once you're in. Full list in [`docs/slash_commands.md`](docs/slash_commands.md).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Layout

```
src/
  config.py        env, models, knobs, paths, HTTP headers
  prompt.py        persona prompt + opener pool
  time_context.py  time-of-day + session-duration system block
  history.py       memory gradient + system-block swap helper
  state.py         internal state: load/save/format/parse/update
  facts.py         user-facts system block formatter
  db.py            SQLite layer: sessions, messages, snapshots, facts
  commands.py      slash-command registry + dispatcher
  chat.py          OpenRouter call: caching, streaming, humanized pacing
  lem.py           CLI entry point
  web.py           FastAPI app (chat, commands, introspection)
  templates/
    index.html     single-page web UI
tests/             pytest suite (one file per src module)
docs/              architecture, slash commands, db schema, web UI, empathy research
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — how all the pieces fit
- [`docs/slash_commands.md`](docs/slash_commands.md) — command reference
- [`docs/web_ui.md`](docs/web_ui.md) — endpoints + SSE protocol
- [`docs/db_schema.md`](docs/db_schema.md) — SQLite tables and access patterns
- [`docs/empathy_research.md`](docs/empathy_research.md) — algorithmic approaches to empathetic dialogue (research, not implemented)
- [`temporal_reasoning.txt`](temporal_reasoning.txt) — original v2→v4 design notes
