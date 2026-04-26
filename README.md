# lemon

A chatbot that simulates a friend, not an assistant. Runs on OpenRouter (Claude Haiku 4.5 by default, with Anthropic-style prompt caching on the persona block). Models both lemon AND the user with the same three-layer internal state (Big 5 traits + characteristic adaptations + PAD core affect — see [`docs/dyadic_state.md`](docs/dyadic_state.md)). Per-turn empathy pipeline: merged pre-reply read (emotion + theory-of-mind + state nudges for both agents in one LLM call) → memory retrieval → draft → sentiment-mirror check → regenerate-once on failure. Facts auto-extract in a backgrounded post-reply call so the user never waits on bookkeeping. Everything persists across sessions in SQLite.

<p align="center">
  <img src="src/static/lemon.png" alt="lemon" width="200" />
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
python -m app.lem               # from src/, or with PYTHONPATH=src
# or:
PYTHONPATH=src python -m app.lem
```

Web UI:
```bash
python -m app.web               # from src/
# or with uvicorn directly:
uvicorn app.web:app --reload --app-dir src
# then visit http://127.0.0.1:8000
```

`pytest.ini` puts `src/` on the import path for the test suite, so tests import the new packages (`app`, `core`, `prompts`, etc.) directly.

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

| method + path        | tag           | what it returns                                                                                       |
| -------------------- | ------------- | ----------------------------------------------------------------------------------------------------- |
| `GET /state`         | introspection | Lemon's three-layer tonic state (traits / adaptations / PAD)                                          |
| `GET /user_state`    | introspection | The user's inferred three-layer tonic state (same schema)                                             |
| `GET /facts`         | introspection | All stored user facts as a key/value map                                                              |
| `GET /sessions`      | introspection | The latest 20 sessions (id, started_at, ended_at, msg_count)                                          |
| `GET /history`       | introspection | Current session's user/assistant turns (system blocks excluded)                                       |
| `GET /trace`         | introspection | Last pipeline trace: emotion read, ToM, memories, empathy-check, facts, regen, both state trajectories |

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

Type `/help` once you're in. Notable ones:
- `/state` / `/user_state` — render either three-layer state object (mood, PAD, traits, adaptations)
- `/why` — pipeline trace for the last reply (emotion read, ToM, memories, post-check, both state trajectories)
- `/empathy on|off`, `/autofacts on|off`, `/cache on|off` — runtime toggles
- `/search`, `/recall`, `/stats`, `/config` — introspection helpers

Full list in [`docs/slash_commands.md`](docs/slash_commands.md).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Layout

```
src/
  app/                   entry points + per-turn orchestration
    lem.py               CLI entry point (python -m app.lem)
    web.py               FastAPI app (chat, commands, introspection, /state, /user_state, /trace)
    pipeline.py          orchestrates read_user (4 sub-objects) → memory → draft → check → regen-once
    session_context.py   shared CLI+web helpers: initial history, refresh blocks, bg fact bookkeeping
    commands.py          slash-command registry + dispatcher; ChatContext (history, lemon_state, user_state)

  core/                  cross-cutting infrastructure
    config.py            env, models, knobs, paths, HTTP headers
    logging_setup.py     `lemon.*` logger tree, payload-safe formatters

  prompts/               every prompt + every block formatter
    __init__.py          format_lemon_state, format_user_state_block, format_reading_block, build_user_read_prompt, build_bookkeep_prompt, etc.
    persona.py           LEMON_TRAITS (Big 5) + LEMON_ADAPTATIONS (goals/values/concerns/stance)
    prompt_stack.py      replace_system_block + compress_history
    schwartz.py          Schwartz's 10 universal values + alias coercion + entry normalizer

  empathy/               the empathy pipeline's specialised building blocks
    emotion.py           23-label emotion schema, family map, validator
    tom.py               theory-of-mind schema, validator
    fact_extractor.py    fact-key regex + value-hygiene validator
    empathy_check.py     sentiment-mirror post-check (regex, 12 detectors)
    user_read.py         merged pre-gen LLM call: emotion + ToM + user_delta + lemon_delta
    post_exchange.py     post-gen LLM call: facts only (state moved pre-reply in stage 2)

  llm/                   raw LLM wire + parsing helpers
    chat.py              OpenRouter reply call: prompt caching + SSE streaming
    parse_utils.py       shared fence-stripper + recent-msgs prompt formatter

  storage/               persistence + retrieval
    db.py                SQLite: sessions, messages, lemon_state_snapshots, user_state_snapshots, facts
    memory.py            composite-scored message retrieval (FTS5 + recency + intensity + emotion)
    lemon_state.py       lemon's three-layer state: defaults, load/save, validator, legacy migrator
    user_state.py        user's three-layer state: defaults, load/save, validator, apply_delta
    state.py             DEPRECATED legacy 6-field shim (kept for migration path)

  temporal/              time-context helpers (humanize_age, time_of_day, session_duration)

  templates/             single-page web UI (index.html)
  static/                brand asset (lemon.png)

tests/                   pytest suite
docs/                    architecture, dyadic_state, memory_architecture, slash_commands, db_schema, web_ui, empathy_research, TECHNICAL, BENCHMARKING
```

Per-turn LLM cost: **3 calls** (`user_read` + reply + facts-only `post_exchange`), of which only the first two block the user. Retry on sentiment-mirror failure adds a fourth reply call. Stage 2 of the dyadic-state architecture kept the round-trip count constant by folding the lemon-state nudge into the existing user_read call.

## Documentation

- [`docs/system_overview.md`](docs/system_overview.md) — **start here**. Whole-system tour with the per-turn flow diagram, module-by-module breakdown, frameworks list, and how every piece connects to the project's goal.
- [`docs/TECHNICAL.md`](docs/TECHNICAL.md) — full technical reference: every module, every call path, every system block, every knob
- [`docs/architecture.md`](docs/architecture.md) — how all the pieces fit, per-turn pipeline diagram
- [`docs/dyadic_state.md`](docs/dyadic_state.md) — three-layer dyadic state design (Big 5 + adaptations + PAD), psychological grounding, schema, migration history
- [`docs/memory_architecture.md`](docs/memory_architecture.md) — three-tier memory (facts / episodic / working history), composite scoring formula, FTS5 setup
- [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) — how to evaluate lemon: ceiling vs stack-lift tests, EQ-Bench 3 / HEART / CES-LCC recipes, pipeline ON vs OFF
- [`docs/slash_commands.md`](docs/slash_commands.md) — command reference (`/help`, `/state`, `/user_state`, `/why`, `/empathy`, ...)
- [`docs/web_ui.md`](docs/web_ui.md) — endpoints + SSE protocol
- [`docs/db_schema.md`](docs/db_schema.md) — SQLite tables and access patterns
- [`docs/empathy_research.md`](docs/empathy_research.md) — survey of algorithmic empathy techniques (the basis of the pipeline)
- [`temporal_reasoning.txt`](temporal_reasoning.txt) — original v2→v4 design notes
