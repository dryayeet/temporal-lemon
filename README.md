# lemon

A chatbot that simulates a friend, not an assistant. Runs on OpenRouter (Claude Haiku 4.5 by default, with Anthropic-style prompt caching on the persona block). Keeps an internal emotional state, runs a per-turn empathy pipeline (merged emotion + theory-of-mind read → memory retrieval → draft → sentiment-mirror check → regenerate-once on failure), and auto-extracts durable facts + nudges its own state in a single post-reply bookkeeping call that runs in the background so the user never waits on it. Everything persists across sessions in SQLite.

<p align="center">
  <img src="src/static/lemon.png" alt="lemon" width="190" />
</p>


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
| `LEMON_AUTO_FACTS`     | `1`                              | auto-extract durable facts from each exchange                  |
| `LEMON_AUTO_FACTS_MAX` | `3`                              | max facts saved per turn                                       |
| `LEMON_DB`             | `.lemon.db`                      | SQLite path (resolved against project root; absolute OK)       |

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

## HTTP API

Once `python src/web.py` is running, the FastAPI app exposes:

**Interactive docs**

| path             | what it serves                                  |
| ---------------- | ----------------------------------------------- |
| `/docs`          | Swagger UI (try requests in-browser)            |
| `/redoc`         | ReDoc (read-only, prettier reference)           |
| `/openapi.json`  | Raw OpenAPI 3 spec                              |

**Health & liveness**

| method + path | tag    | what it returns                                                                                                  |
| ------------- | ------ | ---------------------------------------------------------------------------------------------------------------- |
| `GET /ping`   | health | `{"pong": true}` — trivial liveness probe, returns immediately                                                   |
| `GET /health` | health | DB readiness (`SELECT 1`), uptime, current session id, and the active config snapshot (chat/state model, knobs)  |

**Chat**

| method + path   | tag  | what it does                                                                                                                |
| --------------- | ---- | --------------------------------------------------------------------------------------------------------------------------- |
| `GET /`         | chat | Bundled single-page chat UI                                                                                                 |
| `POST /chat`    | chat | Streams a reply via Server-Sent Events. Event types: `phase`, `token`, `done`, `error`. Body: `{"message": "..."}`         |
| `POST /command` | chat | Runs a slash command (`/help`, `/facts`, `/reset`, …). Body: `{"text": "/help"}` → `{"output": "...", "exit": false}`       |

**Introspection** (read-only)

| method + path     | tag           | what it returns                                                                          |
| ----------------- | ------------- | ---------------------------------------------------------------------------------------- |
| `GET /state`      | introspection | The 6-field internal state dict                                                          |
| `GET /facts`      | introspection | All stored user facts as a key/value map                                                 |
| `GET /sessions`   | introspection | The latest 20 sessions (id, started_at, ended_at, msg_count)                             |
| `GET /history`    | introspection | Current session's user/assistant turns (system blocks excluded)                          |
| `GET /trace`      | introspection | Last pipeline trace: emotion read, ToM, retrieved memories, empathy-check, facts, regen  |

**Static assets**

| method + path           | what it serves                       |
| ----------------------- | ------------------------------------ |
| `GET /favicon.ico`      | The lemon icon (PNG, 24h cached)     |
| `GET /static/lemon.png` | Same icon used by the in-page header |

Every request also returns a correlation id in the `X-Request-ID` response header; the same id appears in the server log lines for that request.

### Logging

The server (and CLI) configure a `lemon.*` logger tree at startup. Tune via env:

| variable            | default | meaning                                                              |
| ------------------- | ------- | -------------------------------------------------------------------- |
| `LEMON_LOG_LEVEL`   | `INFO`  | `DEBUG` dumps full request/response bodies and prompts; `INFO` shows every API call, response, DB write, pipeline phase |
| `LEMON_LOG_FILE`    | (unset) | If set, also writes the same records to this path                    |

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
  config.py              env, models, knobs, paths, HTTP headers
  pipeline.py            orchestrates read_user → memory → draft → check → regen-once
  session_context.py     shared CLI+web helpers: initial history, refresh blocks, bg bookkeeping
  commands.py            slash-command registry + dispatcher
  lem.py                 CLI entry point
  web.py                 FastAPI app (chat, commands, introspection, /trace)

  prompt/                system-prompt content lemon reads
    persona.py           the persona prompt + opener pool
    time_context.py      time-of-day + session-duration block
    history.py           memory gradient + system-block swap helper
    facts.py             <user_facts> system block formatter

  empathy/               the empathy pipeline's specialised building blocks
    emotion.py           emotion schema, validator, system-block formatter
    tom.py               theory-of-mind schema, validator, system-block formatter
    fact_extractor.py    fact-key regex + value-hygiene validator
    empathy_check.py     sentiment-mirror post-check (regex, 12 detectors)
    user_read.py         merged pre-gen LLM call: emotion + ToM in one round-trip
    post_exchange.py     merged post-gen LLM call: fact extraction + state nudge

  llm/                   raw LLM wire + parsing helpers
    chat.py              OpenRouter reply call: prompt caching + SSE streaming
    parse_utils.py       shared fence-stripper + recent-msgs prompt formatter

  storage/               persistence + retrieval
    db.py                SQLite: sessions, messages (with emotion fields), snapshots, facts
    memory.py            emotion-tagged message retrieval
    state.py             internal state: defaults, load/save/format + validator

  templates/
    index.html           single-page web UI

tests/                   pytest suite (most src modules)
docs/                    architecture, slash commands, db schema, web UI, empathy research
```

Per-turn LLM cost: **3 calls** (`user_read` + reply + `post_exchange`), of which only the first two block the user — bookkeeping runs in a daemon thread after the reply ships. Retry on sentiment-mirror failure adds a fourth reply call.

## Documentation

- [`docs/TECHNICAL.md`](docs/TECHNICAL.md) — full technical reference: every module, every call path, every system block, every knob
- [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) — how to evaluate lemon: ceiling vs stack-lift tests, EQ-Bench 3 / HEART / CES-LCC recipes, pipeline ON vs OFF
- [`docs/architecture.md`](docs/architecture.md) — how all the pieces fit, per-turn pipeline diagram
- [`docs/slash_commands.md`](docs/slash_commands.md) — command reference (`/help`, `/why`, `/empathy`, ...)
- [`docs/web_ui.md`](docs/web_ui.md) — endpoints + SSE protocol
- [`docs/db_schema.md`](docs/db_schema.md) — SQLite tables and access patterns
- [`docs/empathy_research.md`](docs/empathy_research.md) — survey of algorithmic empathy techniques (the basis of the pipeline)
- [`temporal_reasoning.txt`](temporal_reasoning.txt) — original v2→v4 design notes
