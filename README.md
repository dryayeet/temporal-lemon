# lemon

A chatbot that simulates a friend, not an assistant. Runs on OpenRouter (Claude Haiku 4.5 by default, with Anthropic-style prompt caching on the persona block). Keeps an internal emotional state, runs a per-turn empathy pipeline (emotion classifier → memory → theory-of-mind → sentiment-mirror check), remembers facts about you across sessions in SQLite, and types like a person.

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

| variable               | default                          | meaning                                                        |
| ---------------------- | -------------------------------- | -------------------------------------------------------------- |
| `LEMON_CHAT_MODEL`     | `anthropic/claude-haiku-4.5`     | main chat model                                                |
| `LEMON_STATE_MODEL`    | `anthropic/claude-haiku-4.5`     | small model used by the state updater + empathy classifiers    |
| `LEMON_PROMPT_CACHE`   | auto (on for `anthropic/*` only) | toggle Anthropic-style `cache_control` blocks                  |
| `LEMON_EMPATHY`        | `1`                              | enable the empathy pipeline                                    |
| `LEMON_EMPATHY_RETRY`  | `1`                              | regenerate once when the post-check trips                      |
| `LEMON_MEMORY_LIMIT`   | `3`                              | how many past matching-emotion messages to surface per turn    |
| `LEMON_HUMANIZE`       | `1`                              | toggle per-token typing pacing                                 |
| `LEMON_DB`             | `.lemon.db`                      | SQLite path (gitignored)                                       |

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

Type `/help` once you're in. Notable additions:
- `/empathy on|off` — toggle the empathy pipeline live
- `/why` — show the pipeline trace for the last reply (emotion read, ToM, post-check result)

Full list in [`docs/slash_commands.md`](docs/slash_commands.md).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Layout

```
src/
  config.py         env, models, knobs, paths, HTTP headers
  prompt.py         persona prompt + opener pool
  time_context.py   time-of-day + session-duration system block
  history.py        memory gradient + system-block swap helper
  state.py          internal state: load/save/format/parse/update
  facts.py          user-facts system block formatter
  db.py             SQLite: sessions, messages (with emotion fields), snapshots, facts
  commands.py       slash-command registry + dispatcher
  chat.py           OpenRouter call: caching, streaming, humanized pacing
  emotion.py        pre-generation user-emotion classifier
  tom.py            theory-of-mind side pass
  memory.py         emotion-tagged retrieval from db
  empathy_check.py  sentiment-mirror post-check (regex / heuristics)
  pipeline.py       orchestrates classify → retrieve → ToM → draft → check → regen
  lem.py            CLI entry point
  web.py            FastAPI app (chat, commands, introspection, /trace)
  templates/
    index.html      single-page web UI
tests/              pytest suite (one file per src module)
docs/               architecture, slash commands, db schema, web UI, empathy research
```

## Documentation

- [`docs/TECHNICAL.md`](docs/TECHNICAL.md) — full technical reference: every module, every call path, every system block, every knob
- [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) — how to evaluate lemon: ceiling vs stack-lift tests, EQ-Bench 3 / HEART / CES-LCC recipes, pipeline ON vs OFF
- [`docs/architecture.md`](docs/architecture.md) — how all the pieces fit, per-turn pipeline diagram
- [`docs/slash_commands.md`](docs/slash_commands.md) — command reference (`/help`, `/why`, `/empathy`, ...)
- [`docs/web_ui.md`](docs/web_ui.md) — endpoints + SSE protocol
- [`docs/db_schema.md`](docs/db_schema.md) — SQLite tables and access patterns
- [`docs/empathy_research.md`](docs/empathy_research.md) — survey of algorithmic empathy techniques (the basis of the pipeline)
- [`temporal_reasoning.txt`](temporal_reasoning.txt) — original v2→v4 design notes
