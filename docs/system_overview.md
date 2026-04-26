# Lemon, end to end

Lemon is a chatbot that tries to feel like a friend instead of an assistant.
It runs as either a CLI or a small web app, persists everything to a single
SQLite file on your machine, and talks to one external service: an LLM
through OpenRouter. That's the whole stack.

If you're new here, read this top to bottom. For deeper drill-downs the
links live at the bottom.

---

## 1. The idea

Most chatbots are shaped like "user asks, bot answers." Lemon is shaped
differently:

- **Two people are in the room.** Lemon has her own state. You have your
  own state. Both are tracked and both update every turn.
- **State first, reply second.** What lemon "feels" about your message
  shapes the reply, not the other way around.
- **Memory persists.** Facts, mood drift, and things you mentioned last
  Tuesday all survive across sessions.
- **Texting register, not therapy register.** No unsolicited advice, no
  clinical labels, no "I hear you, that's so valid" stacking.

Everything else exists in service of those four points.

---

## 2. The shape of the system

```
                ┌──────────────────────────────────────────────┐
                │   You (chat msg, or /slash command)          │
                └──────────────┬───────────────────────────────┘
                               │
                ┌──────────────▼───────────────┐
                │  app/                        │
                │   lem.py     (CLI)           │
                │   web.py     (FastAPI + SSE) │
                │   pipeline.py (orchestrator) │
                │   session_context, commands  │
                └──────────────┬───────────────┘
                               │
   ┌─────────────┬─────────────┼─────────────┬──────────────┐
   ▼             ▼             ▼             ▼              ▼
 prompts/      empathy/        llm/        storage/       temporal/
 (every      (read user,    (OpenRouter   (SQLite +      (time of day,
  prompt +    critique,      via          state +         age decay)
  formatter,  extract        requests)    memory)
  Schwartz)   facts)
                                            │
                                            ▼
                            ┌────────────────────────────────┐
                            │ .lemon.db   (single SQLite file)│
                            │  6 tables + 1 FTS5 virtual      │
                            └────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────┐
  │  core/   config (env, knobs)  +  logging_setup             │
  │          imported by everyone                              │
  └────────────────────────────────────────────────────────────┘
```

Every Python file lives inside one of those folders. `app/` is the front
door, the rest are workers it calls into.

---

## 3. What happens when you send a message

```mermaid
flowchart TD
    A[You hit send] --> B{slash command?}
    B -- "/cmd" --> C[commands.dispatch] --> END[done]

    B -- "regular text" --> D[refresh time + lemon_state + facts blocks]
    D --> E[user_read.read_user<br/>1 LLM call]

    E -. emits .-> E1[your emotion]
    E -. .-> E2[theory of mind on you]
    E -. .-> E3[your state delta]
    E -. .-> E4[lemon's state delta]

    E --> F[apply both deltas, persist both states]
    F --> G[memory.relevant_memories<br/>FTS5 + scoring, no LLM]
    G --> H[inject system blocks<br/>lemon_state, user_state, reading, memory]
    H --> I[generate the reply<br/>1 LLM call, buffered]
    I --> J[empathy_check<br/>12 regex detectors, no LLM]
    J -- pass --> K[ship reply]
    J -- fail --> L[regenerate once<br/>with critique<br/>1 LLM call]
    L --> K
    K --> M[background: bookkeep<br/>1 LLM call → save facts]
    M --> END
```

**Two things wait on the user:** the read pass and the reply itself.
The fact-extraction call fires after you've already seen the reply, in a
background thread. That's three LLM calls per turn (four if the empathy
check fails and a regenerate happens). All are Haiku 4.5 by default.

The non-obvious bit is **state-first ordering**. Both states get updated
during the read pass. Then the reply is generated against those
just-updated states. So when lemon replies, her tone reflects what she
read, not what she carried in.

---

## 4. The folders, briefly

### `app/`

Where a turn actually runs.

- **`pipeline.py`** owns `run_empathy_turn()`, the thing that wires every
  step together. The graph calls it the biggest cross-community bridge,
  which is just a fancy way of saying it touches almost everything.
- **`lem.py`** and **`web.py`** are the two entry points. CLI is a stdin
  loop; web is FastAPI plus Server-Sent Events.
- **`session_context.py`** has helpers both entry points share:
  building the initial system-prompt stack, refreshing per-turn blocks,
  running the post-reply bookkeeping thread.
- **`commands.py`** is the slash-command registry. ~22 commands, same
  dispatcher for CLI and web.

### `core/`

Two files, both imported everywhere.

- **`config.py`** has every env-tunable knob (model IDs, toggles, memory
  weights, DB path). Loads `.env` at startup.
- **`logging_setup.py`** sets up a `lemon.*` logger tree with two
  payload-safe formatters: `preview()` for one-liners, `shape_of()`
  for "what's the structure" without leaking content.

### `prompts/`

Every prompt and every block of text lemon ever sees.

- **`__init__.py`** is the big one (~900 lines). The persona prompt,
  the 23-label emotion vocabulary, every `format_*_block` function,
  and the two prompt builders for `user_read` and `bookkeep`.
- **`persona.py`** holds lemon's static identity: a Big 5 trait vector
  and a small adaptations dict (goals, values tagged with Schwartz
  categories, concerns, relational stance).
- **`prompt_stack.py`** is two helpers: `replace_system_block()` (swap a
  tagged block in place) and `compress_history()` (keep the last 8 turns
  verbatim, fold older ones into one summary block).
- **`schwartz.py`** is Schwartz's 10 universal values and a normalizer.
  Used to tag every entry in the `values` slot.

### `empathy/`

The "read the user, critique the draft, extract facts" stack.

- **`emotion.py`** has the 23-label emotion schema and the family map
  (sad / anger / fear / self_conscious / positive / etc.) used by
  memory retrieval.
- **`tom.py`** has the theory-of-mind schema (`feeling`, `avoid`,
  `what_helps`).
- **`user_read.py`** is the merged pre-reply LLM call. It returns four
  things: emotion, ToM, your state delta, lemon's state delta. A small
  helper inside it (`_clamp_lemon_delta`) makes lemon's deltas tighter
  than yours so she stays stable while you can swing.
- **`post_exchange.py`** is the post-reply LLM call. Just facts now;
  state moved pre-reply.
- **`fact_extractor.py`** is the validator and dedup gate that stops the
  LLM from inventing key mutations like `sleep_status_v2`.
- **`empathy_check.py`** is 12 regex detectors that run on the draft
  before it ships. Catches minimizing, toxic positivity, advice-pivot,
  validation cascades, therapy-speak, and a handful more. If anything
  trips, the pipeline regenerates once with the critique.

### `llm/`

The wire layer.

- **`chat.py`** does the OpenRouter call via `requests`, with
  Anthropic-style `cache_control` on the persona block (effectively free
  after turn 1 on Anthropic models).
- **`parse_utils.py`** has two utilities every classifier uses:
  `strip_json_fences()` and a recent-messages formatter for the prompts.

### `storage/`

SQLite plus a few thin wrappers around it.

- **`db.py`** has the schema, migrations, and CRUD helpers. Six tables
  plus an FTS5 virtual table for full-text search. Every helper opens a
  short-lived connection through a context manager.
- **`memory.py`** does the per-turn retrieval. FTS5 picks candidates,
  then a four-signal score (lexical + recency + intensity +
  emotion-family) re-ranks them.
- **`lemon_state.py`** and **`user_state.py`** hold the three-layer state
  for each agent. Same shape, different dynamics.
- **`state.py`** is the deprecated 6-field shim. Kept around so legacy
  snapshots can still be migrated on first read.

### `temporal/`

Two small files that color lemon's tone by clock.

- **`age.py`**: `humanize_age()` for memory-block timestamps,
  `recency_decay()` for memory scoring.
- **`clock.py`**: time-of-day buckets and session-duration notes for
  the `<time_context>` block. After 11pm lemon gets quieter; in the
  morning she's a bit fresher.

### `templates/` + `static/`

One HTML file with inline CSS and vanilla JS. Sidebar shows lemon's full
state, your full state, your facts, and recent sessions, all
auto-refreshing after every turn. Light/dark toggle stored in
`localStorage`.

---

## 5. Frameworks and dependencies

Python 3.12. The list is intentionally short.

| dep | role |
|---|---|
| FastAPI | Web app + auto OpenAPI docs at `/docs`. |
| uvicorn | ASGI server. |
| requests | HTTP to OpenRouter. |
| pydantic | FastAPI request/response models. |
| python-dotenv | Loads `.env` at config import. |
| pytest | Test runner. `pytest.ini` puts `src/` on the path. |
| sqlite3 (stdlib) | Single-file persistence + FTS5. |
| OpenRouter (HTTP) | LLM gateway. Default model: `anthropic/claude-haiku-4.5` for everything (chat + classifiers). |

No vector DB, no embeddings, no ORM, no async framework, no auth. One
process, one user, localhost only.

---

## 6. The four objects you'll see everywhere

**Three-layer state** (lemon and you have one each, same shape):

```python
{
    "traits": {                            # Big 5, each in [-1, +1]
        "openness": ..., "conscientiousness": ..., "extraversion": ...,
        "agreeableness": ..., "neuroticism": ...,
    },
    "adaptations": {                       # mid-term context
        "current_goals":     list[str],
        "values":            list[{"label": str, "schwartz": str | None}],
        "concerns":          list[str],
        "relational_stance": str | None,
    },
    "state": {                             # PAD mood, each in [-1, +1]
        "pleasure": ..., "arousal": ..., "dominance": ...,
        "mood_label": str,
    },
}
```

**Phasic emotion** (this turn only, not stored separately):

```python
{"primary": <one of 23 labels>, "intensity": float,
 "underlying_need": str | None, "undertones": [...]}
```

**`PipelineTrace`** (per turn, on `ctx.last_trace`): every intermediate
output the pipeline produces. Surfaces via `/why` and `GET /trace`.

**Message envelope** (a row in the `messages` table): id, session_id,
role, content, created_at, plus emotion / intensity / salience on user
rows.

---

## 7. What gets persisted

| object | table | written when |
|---|---|---|
| Sessions | `sessions` | start at boot, end at shutdown |
| Every message | `messages` (+ FTS5 mirror) | per turn, both sides |
| Lemon's tonic state | `lemon_state_snapshots` | per turn, pre-reply |
| Your tonic state | `user_state_snapshots` | per turn, pre-reply |
| Facts about you | `facts` (key/value, upsert) | post-reply via `bookkeep`, or `/remember` |
| Old 6-field state | `state_snapshots` | deprecated, read-only archive |

The whole thing is one `.lemon.db` file at the project root. Schema
version 3 right now.

---

## 8. How to run it

```bash
# CLI
PYTHONPATH=src python -m app.lem

# Web (FastAPI + SSE)
PYTHONPATH=src python -m app.web
# or:
uvicorn app.web:app --reload --app-dir src
```

The web server exposes nine HTTP endpoints. The interesting ones:

- `POST /chat` streams replies as Server-Sent Events.
- `POST /command` runs a slash command.
- `GET /state`, `/user_state`, `/facts`, `/sessions`, `/history`, `/trace`
  are read-only introspection.
- `GET /docs` is the auto-generated Swagger UI.

Slash commands work the same in the CLI and the web composer. The full
list lives in [`slash_commands.md`](slash_commands.md).

---

## 9. What it doesn't do

- **Multi-user.** Two browser tabs against the same server share one
  conversation. There's no auth either, so don't expose past localhost.
- **Semantic memory retrieval.** It's BM25 plus a small composite scorer.
  `slept` and `sleep` match (porter stemming), but `exhausted` and
  `wiped out` don't. Adding embeddings is the highest-return future
  work; see `memory_architecture.md`.
- **Real personality inference.** Trait nudges per turn are capped tight
  enough that the user's Big 5 estimate is essentially frozen. A real
  inference pass would aggregate across many sessions offline.
- **Semantic empathy check.** The 12 regex detectors catch known
  failure modes by phrase. Paraphrases of the same failure modes will
  slip through. `empathy_research.md` lists the next steps.

---

## 10. Where to read next

- **[`TECHNICAL.md`](TECHNICAL.md)** is the full reference: every module,
  every call path, every config knob.
- **[`dyadic_state.md`](dyadic_state.md)** is the long version of the
  three-layer state design. Why Big 5 + PAD + Schwartz, why the McAdams
  scaffold, what changed in each migration stage.
- **[`memory_architecture.md`](memory_architecture.md)** is the long
  version of section 3's memory step. The composite scoring formula,
  FTS5 setup, eval harness.
- **[`empathy_research.md`](empathy_research.md)** surveys the
  algorithmic-empathy literature this pipeline is built on, and flags
  what's implemented vs future work.
- **[`slash_commands.md`](slash_commands.md)**, **[`web_ui.md`](web_ui.md)**,
  **[`db_schema.md`](db_schema.md)**, **[`BENCHMARKING.md`](BENCHMARKING.md)**
  are smaller targeted references.
- **[`graphify-out/GRAPH_REPORT.md`](../graphify-out/GRAPH_REPORT.md)** is
  the auto-generated knowledge graph: god nodes, communities, suggested
  questions to dig into. Refresh it with `graphify update .` after any
  layout change.
