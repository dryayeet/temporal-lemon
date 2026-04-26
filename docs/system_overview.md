# Lemon — System Overview

A single-user chat companion designed to simulate a friend, not an assistant.
Built around a per-turn empathy pipeline that reads the user, retrieves
relevant past moments, drafts a reply, runs a sentiment-mirror check, and
extracts durable facts in the background. Both lemon and the user are
modelled with the same three-layer internal state (Big 5 traits +
characteristic adaptations + PAD core affect). Runs as either a CLI REPL
or a FastAPI web app, persisting everything to one local SQLite file.

This doc is the "whole system at a glance" view. For deeper drill-downs see
[`TECHNICAL.md`](TECHNICAL.md), [`dyadic_state.md`](dyadic_state.md),
[`memory_architecture.md`](memory_architecture.md), and
[`empathy_research.md`](empathy_research.md).

---

## 1. The project's goal

Most chatbots present as helpful assistants. Lemon's premise is different:
**simulate a real friend**. That means:

- Two interacting agents, **both** with persistent internal state — not
  user-as-input + bot-as-output.
- Reply generation is downstream of state. What lemon "feels" in response
  to your message shapes how she replies, not the other way around.
- Memory persists across sessions: facts, mood trajectories, things
  quietly carried in.
- The tone is texting-with-a-friend, not therapy-session-transcript. No
  unsolicited advice, no clinical labelling, no validation cascade.

Everything else in the system is in service of that goal.

---

## 2. High-level architecture

```
                ┌──────────────────────────────────────────────┐
                │  User input (chat msg or /slash command)    │
                └──────────────┬───────────────────────────────┘
                               │
                ┌──────────────▼───────────────┐
                │  app/  — entry + orchestrate │
                │   lem.py / web.py            │
                │   pipeline.py                │
                │   session_context.py         │
                │   commands.py                │
                └──────────────┬───────────────┘
                               │
   ┌─────────────┬─────────────┼─────────────┬──────────────┐
   ▼             ▼             ▼             ▼              ▼
┌────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐
│prompts/│  │empathy/ │  │   llm/   │  │ storage/ │  │ temporal/  │
│        │  │         │  │          │  │          │  │            │
│ blocks │  │ user_   │  │ chat.py  │  │  db.py   │  │ age,clock  │
│ persona│  │  read   │  │ (Open-   │  │ +memory  │  │            │
│ schwartz│ │ post_   │  │ Router,  │  │ +states  │  │            │
│ stack  │  │  exch   │  │ requests)│  │ (SQLite) │  │            │
│        │  │ check   │  │          │  │          │  │            │
└────────┘  └─────────┘  └──────────┘  └──────────┘  └────────────┘
                                            │
                                            ▼
                            ┌────────────────────────────────┐
                            │ .lemon.db  (single SQLite file)│
                            │  6 tables + 1 FTS5 virtual     │
                            └────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────┐
  │  core/  — config (env, models, knobs) + logging_setup      │
  │           imported by every module above                   │
  └────────────────────────────────────────────────────────────┘
```

Every Python file lives in a folder. Reads top-down: user input enters
through `app/`, which orchestrates calls into `prompts/`, `empathy/`, `llm/`,
`storage/`, and `temporal/`. State and memory both round-trip through
SQLite. `core/` is cross-cutting infrastructure imported by everyone.

---

## 3. Per-turn flow

Showing what happens when you type a message and hit send.

```mermaid
flowchart TD
    A[User sends message] --> B{slash command?}
    B -- "/cmd" --> C[commands.dispatch]
    C --> CR[render system bubble]
    CR --> END[Done]

    B -- "regular text" --> D[refresh_base_blocks<br/>time + lemon_state + facts]
    D --> E[user_read.read_user<br/>1 LLM call]

    E -. emits 4 sub-objects .-> E1[emotion]
    E -. .-> E2[theory_of_mind]
    E -. .-> E3[user_state_delta]
    E -. .-> E4[lemon_state_delta]

    E --> F[apply both deltas<br/>→ persist user_state<br/>→ persist lemon_state]
    F --> G[memory.relevant_memories<br/>FTS5 + composite score, no LLM]
    G --> H[Inject system blocks:<br/>lemon_state user_state reading memory]
    H --> I[llm.generate_reply<br/>1 LLM call → buffered draft]
    I --> J[empathy_check.check_response<br/>12 regex detectors, no LLM]
    J -- pass --> K[Ship reply]
    J -- fail --> L[regenerate-once<br/>with critique<br/>1 LLM call]
    L --> K
    K --> M[Background: post_exchange.bookkeep<br/>1 LLM call → facts only]
    M --> END
```

**LLM cost per turn:** 2 user-blocking calls (`user_read` + `generate_reply`)
plus 1 backgrounded (`bookkeep`). Add 1 more if the empathy check fails.
With `/empathy off`: just `generate_reply` blocks the user.

**State-first ordering:** both agents' tonic states are nudged *before* the
reply, in the merged `user_read` pass. The reply call reads `<lemon_state>`
already containing the post-nudge state, so reply tone reflects what lemon
"feels" in response to what was just said.

---

## 4. Modules

### `app/` — entry + orchestration

The five files that drive a turn end-to-end.

| file | purpose |
|---|---|
| `lem.py` | CLI REPL entry point. Run as `python -m app.lem`. Reads stdin, dispatches `/commands` or runs the pipeline, prints replies. |
| `web.py` | FastAPI app + Server-Sent Events streaming. Run as `python -m app.web` or `uvicorn app.web:app --app-dir src`. Single-user, single-process, localhost-only. |
| `pipeline.py` | The orchestrator: `run_empathy_turn()`. The graph's biggest cross-community bridge (39 edges to 10 different module clusters). Owns the `PipelineTrace` dataclass that flows through `/why`. |
| `session_context.py` | Shared CLI+web helpers — `initial_history`, `refresh_base_blocks` (rebuilds `<time_context>`, `<lemon_state>`, `<user_facts>` between turns), and `run_bookkeeping` (the daemon-thread fact extractor that fires after the reply ships). |
| `commands.py` | Slash-command registry + `ChatContext` dataclass. ~22 commands (`/help`, `/state`, `/user_state`, `/why`, `/search`, etc.) registered via a `@command(name, help)` decorator. Same dispatcher serves both the CLI and the web `/command` HTTP endpoint. |

### `core/` — cross-cutting infrastructure

| file | purpose |
|---|---|
| `config.py` | All env-tunable knobs in one place: model IDs, prompt-cache toggle, empathy-pipeline toggle, memory-scoring weights, DB path, OpenRouter HTTP headers. Loads `.env` via `python-dotenv` at import time. |
| `logging_setup.py` | The `lemon.*` logger tree. Two payload-safe formatters: `preview()` for one-line message snippets and `shape_of()` for structure-without-content. Default level INFO; flip to DEBUG via `LEMON_LOG_LEVEL=DEBUG`. |

### `prompts/` — every prompt + every system-block formatter

| file | purpose |
|---|---|
| `__init__.py` | The canonical big module (~900 lines, formerly `prompts.py`). Defines `LEMON_PROMPT` (the persona system block), `EMOTION_LABELS` (23-label vocabulary), every `*_TAG` constant, every `format_*_block()` function, and the two LLM-prompt builders `build_user_read_prompt` and `build_bookkeep_prompt`. |
| `persona.py` | `LEMON_TRAITS` (Big 5 calibrated against `LEMON_PROMPT`) and `LEMON_ADAPTATIONS` (goals, Schwartz-tagged values, concerns, relational stance) — the static seeds of lemon's three-layer state. |
| `prompt_stack.py` | `replace_system_block()` (drop-and-reinsert by tag) and `compress_history()` (memory gradient: keep last 8 turns verbatim, fold older ones into one `<earlier_conversation>` block). |
| `schwartz.py` | Schwartz's 10 universal values (Schwartz 1992) + alias coercion + `normalize_value_entry()`. Used to tag entries in the `values` slot. |

### `empathy/` — read, critique, extract

| file | purpose |
|---|---|
| `emotion.py` | 23-label phasic emotion schema, family map (sad / anger / fear / self-conscious / positive / etc.), validator. Family map drives the mood-congruence boost in memory retrieval. |
| `tom.py` | Theory-of-mind schema (`feeling` / `avoid` / `what_helps`) and validator. |
| `user_read.py` | The merged pre-gen LLM call. Returns 4 sub-objects: emotion + ToM + user_state_delta + lemon_state_delta. Asymmetric clamping (`_clamp_lemon_delta`) enforces lemon's tighter dynamics (PAD ±0.10 vs user's ±0.15; lemon's traits and values frozen). |
| `post_exchange.py` | The post-gen LLM call. Facts only (state moved pre-reply in stage 2 of the dyadic-state work). Runs in a daemon thread so the user never waits on it. |
| `fact_extractor.py` | Fact-key regex + value-hygiene validator + dedup gate that prevents the LLM from inventing key mutations like `_v2` / `_updated`. |
| `empathy_check.py` | 12 regex detectors run against the draft. Pure Python, no LLM. Detects minimizing, toxic positivity, advice-pivot, validation cascade, therapy-speak, sycophancy, etc. The graph's most-connected node (`check_response`, 50 edges). On failure, the pipeline regenerates once with the critique appended. |

### `llm/` — wire layer

| file | purpose |
|---|---|
| `chat.py` | OpenRouter HTTP call via `requests`. Anthropic-style `cache_control: ephemeral` breakpoint on the persona block (cached free after turn 1 on Anthropic models). Two entry points: `iter_chat` (streaming generator) and `generate_reply` (buffered, used by the pipeline since the post-check needs a complete draft). |
| `parse_utils.py` | `strip_json_fences()` (tolerant of ```json fences) and `format_recent_for_prompt()` (renders the last ~6 non-system turns for classifier prompts). |

### `storage/` — persistence + retrieval

| file | purpose |
|---|---|
| `db.py` | SQLite layer. Six tables (`sessions`, `messages`, `state_snapshots` [legacy archive], `lemon_state_snapshots`, `user_state_snapshots`, `facts`) + one FTS5 virtual table (`messages_fts`, porter stemmer). Idempotent schema, three additive migrations versioned via a `schema_version` table. Every helper is a short-lived `@contextmanager connect()` block. |
| `memory.py` | Composite-scored retrieval. Pulls candidates via FTS5 then re-ranks with a four-signal score: lexical (sigmoid-normalized BM25) + recency (half-life) + intensity + emotion-relatedness (family-aware). Falls back to recent-only when FTS yields nothing. |
| `lemon_state.py` | Lemon's three-layer state: defaults (built from `persona`), session-start re-pegging (PAD baseline, persona-baseline relational stance + values), legacy-shape migrator that converts old 6-field `state_snapshots` rows on first load. |
| `user_state.py` | User's three-layer state. Same shape as lemon's; different dynamics (no session-start overrides — users carry in whatever they had). Defines `validate_delta` and `apply_delta`, shared by both agents. |
| `state.py` | DEPRECATED 6-field shim. Kept only for the legacy migration path. New code shouldn't touch it. |

### `temporal/` — time helpers

| file | purpose |
|---|---|
| `age.py` | `humanize_age()`: `today` / `yesterday` / `N days ago` / `N weeks ago` rendering for memory blocks. Plus `recency_decay()` (half-life weight) used by the memory scorer. |
| `clock.py` | `time_of_day_label()` (morning / afternoon / late night / very late night buckets) and `session_duration_note()` for the `<time_context>` block. Drives lemon's time-aware tone (low-key after 11pm, fresher in the morning). |

### `templates/` + `static/`

Vanilla JS single-page chat UI in `templates/index.html`. No build step.
Sidebar shows lemon's full state, user's full state, facts, and recent
sessions — all auto-refreshed after every turn and every slash command.
Light/dark theme toggle via CSS custom properties; preference stored in
`localStorage`.

---

## 5. Frameworks & dependencies

Python 3.12. The dependency surface is intentionally small.

| dep | role |
|---|---|
| **FastAPI** | Web server (`web.py`). Auto-generates OpenAPI docs at `/docs` and `/redoc`. |
| **uvicorn** | ASGI server for the FastAPI app. |
| **requests** (2.32.5) | Synchronous HTTP client used for OpenRouter calls (`llm/chat.py`, `empathy/user_read.py`, `empathy/post_exchange.py`). |
| **pydantic** (2.9.2) | Request/response models for the FastAPI endpoints. |
| **python-dotenv** (1.2.2) | Loads `.env` at config import time. |
| **pytest** (8.3.3) | Test runner. `pytest.ini` sets `pythonpath = src` so tests import `app.*`, `core.*`, `prompts.*` directly. |
| **sqlite3** (stdlib) | Local single-file persistence + FTS5 + external-content tables + triggers for FTS sync. |
| **OpenRouter** (HTTP API) | LLM gateway. Default model: `anthropic/claude-haiku-4.5` for both chat and auxiliary calls. Anthropic-style prompt caching is enabled by default for `anthropic/*` models. |

No vector DB, no embeddings, no async framework, no ORM, no auth layer.
Single user, single process, localhost-only.

---

## 6. Data shapes

The system has four canonical objects you'll see referenced everywhere.

**Three-layer state** (one per agent — same schema for lemon and user):

```python
{
    "traits": {                        # Big 5 / OCEAN, each in [-1, +1]
        "openness": float, "conscientiousness": float, "extraversion": float,
        "agreeableness": float, "neuroticism": float,
    },
    "adaptations": {
        "current_goals":     list[str],
        "values":            list[{"label": str, "schwartz": str | None}],
        "concerns":          list[str],
        "relational_stance": str | None,
    },
    "state": {                         # PAD core affect, each in [-1, +1]
        "pleasure": float, "arousal": float, "dominance": float,
        "mood_label": str,             # derived; folksy whitelist
    },
}
```

**Phasic emotion event** (per turn, ephemeral, separate from state):

```python
{
    "primary":         "joy" | "sadness" | "tired" | ... (23 labels),
    "intensity":       float,    # [0, 1]
    "underlying_need": str | None,
    "undertones":      list[str],
}
```

**`PipelineTrace`** (one per turn, attached to `ctx.last_trace`):

```python
emotion, tom, draft, check, regenerated, final, memories, facts_extracted,
user_state_before, user_state_after, user_state_delta,
lemon_state_before, lemon_state_after, lemon_state_delta
```

Surfaces via `/why` (slash command) and `GET /trace` (HTTP endpoint).

**Message envelope** (a row in `messages`):

```python
{id, session_id, role, content, created_at, emotion, intensity, salience}
```

Emotion / intensity / salience populated only on user rows by the
pre-reply read; assistant rows leave them NULL.

---

## 7. State persistence

| object | table | when written |
|---|---|---|
| Session boundaries | `sessions` | start_session at boot, end_session at shutdown |
| Every message | `messages` (+ `messages_fts` via triggers) | per turn, both user and assistant |
| Lemon's tonic state | `lemon_state_snapshots` | per turn, **pre-reply** (in pipeline) |
| User's tonic state | `user_state_snapshots` | per turn, **pre-reply** (in pipeline) |
| Durable user facts | `facts` (key/value, upsert) | post-reply via `bookkeep`, or via `/remember` |
| Legacy 6-field state | `state_snapshots` | DEPRECATED — read-only archive |

Phasic emotion events are not stored as a separate object; they're written
as the `emotion`/`intensity` columns on the user message row.

The whole DB is a single `.lemon.db` file at the project root. Schema is
version 3 (most recent migration added `lemon_state_snapshots` for stage 3
of the dyadic-state work).

---

## 8. External entry points

**Two ways to chat:**

```bash
# CLI
PYTHONPATH=src python -m app.lem

# Web UI (FastAPI + SSE)
PYTHONPATH=src python -m app.web
# or:
uvicorn app.web:app --reload --app-dir src
```

**HTTP API surface** (web only):

| method + path | tag | what it does |
|---|---|---|
| `GET /` | chat | Bundled single-page UI |
| `POST /chat` | chat | Streams a reply via SSE (events: `phase`, `token`, `done`, `error`) |
| `POST /command` | chat | Runs a slash command, returns `{output, exit}` |
| `GET /state` | introspection | Lemon's three-layer tonic state |
| `GET /user_state` | introspection | User's three-layer tonic state (same schema) |
| `GET /facts` | introspection | Stored user facts |
| `GET /sessions` | introspection | Last 20 sessions |
| `GET /history` | introspection | Current session's user/assistant turns |
| `GET /trace` | introspection | Last `PipelineTrace` as JSON |
| `GET /ping`, `GET /health` | health | Liveness + readiness probes |
| `GET /docs`, `GET /redoc` | meta | Auto-generated OpenAPI explorers |

**Slash commands** (work in both CLI and web): `/help`, `/state`,
`/user_state`, `/reset`, `/facts`, `/remember`, `/forget`, `/history`,
`/rewind`, `/clear`, `/export`, `/model`, `/sessions`, `/search`,
`/recall`, `/stats`, `/config`, `/empathy`, `/autofacts`, `/cache`, `/why`,
`/quit`. See [`slash_commands.md`](slash_commands.md) for the full reference.

---

## 9. How the pieces relate to the goal

| project goal | how it shows up |
|---|---|
| Two interacting agents, both with state | Three-layer schema in `storage/{lemon,user}_state.py`, persisted in two parallel snapshot tables. Both updated **pre-reply** in one merged LLM call. |
| Reply downstream of state | The pipeline applies both deltas → persists → re-injects `<lemon_state>` block → only then runs `generate_reply`. So the reply call sees freshly-nudged states. |
| Cross-session memory | `messages_fts` (FTS5 with porter stemmer) + composite scoring in `storage/memory.py`. Past user messages from other sessions surface based on lexical relevance + recency + intensity + emotion-family congruence. |
| Texting-not-therapy tone | `LEMON_PROMPT` enforces texting-register voice + "no advice unless asked" + "never name your state". `empathy_check`'s 12 detectors catch common AI-empathy failure modes (minimizing, toxic positivity, validation cascade) and trigger a regenerate-once. |
| Friend persona stability | Lemon's traits + values are persona-fixed (`persona.LEMON_TRAITS`, `persona.LEMON_ADAPTATIONS`). `_clamp_lemon_delta` zeros out trait nudges and tightens her PAD bounds (±0.10 vs user's ±0.15). She can quietly carry a concern about you, but she doesn't mood-mirror. |
| Both agents psychologically grounded | Big 5 (OCEAN) for traits — empirically dominant over MBTI / Enneagram. PAD (Mehrabian-Russell) for state — continuous, used in computational affective agents. Schwartz (1992) for value tagging. McAdams' three-level framework as the integrating scaffold. See [`dyadic_state.md`](dyadic_state.md) §6 for the full grounding. |

---

## 10. Non-goals + known limits

- **Single-user, single-process.** `web.py` keeps one global `ChatContext`.
  Two browser tabs against the same server share one conversation.
- **No auth.** `web.py` is localhost-only by default. Don't expose past
  the loopback without a reverse proxy.
- **Post-check is regex, not semantic.** Paraphrases of minimizing or
  validation-cascade phrases will slip through. Tier-2 ideas (best-of-N,
  multi-agent critic, semantic check) are sketched in `empathy_research.md`.
- **No vector DB.** Memory retrieval is BM25 + composite scoring.
  Catches `sleep / sleeping / slept` via porter stemming but not paraphrases
  (`exhausted` ↔ `wiped out`). Adding embedding similarity is listed as
  the highest-return future work in `memory_architecture.md`.
- **User trait inference is essentially frozen** (per-turn cap of ±0.02).
  Real trait estimation would need an offline aggregation pass over
  many sessions; sketched in `dyadic_state.md` §11.

---

## 11. Where to look next

- **[`TECHNICAL.md`](TECHNICAL.md)** — full module-by-module reference.
  Per-turn lifecycle, every prompt block, every test pattern, every
  config knob.
- **[`dyadic_state.md`](dyadic_state.md)** — the three-layer state design.
  Why Big 5 + PAD + Schwartz, the McAdams scaffold, asymmetric dynamics,
  the three-stage migration from the old 6-field internal_state.
- **[`memory_architecture.md`](memory_architecture.md)** — the three
  memory tiers (facts, episodic, working history), composite scoring
  formula with default weights, FTS5 setup, eval harness.
- **[`empathy_research.md`](empathy_research.md)** — survey of the
  algorithmic-empathy literature this pipeline is built on, with notes on
  what's currently implemented vs Tier-2 / Tier-3 future work.
- **[`slash_commands.md`](slash_commands.md)** — full slash-command
  reference with examples.
- **[`db_schema.md`](db_schema.md)** — column-by-column SQLite reference.
- **[`web_ui.md`](web_ui.md)** — FastAPI endpoints + SSE protocol +
  frontend layout.
- **[`BENCHMARKING.md`](BENCHMARKING.md)** — how to evaluate lemon's
  empathic quality (EQ-Bench 3, HEART, CES-LCC, ablations).
- **[`graphify-out/GRAPH_REPORT.md`](../graphify-out/GRAPH_REPORT.md)** —
  the auto-generated knowledge graph: god nodes, communities, suggested
  questions. Refresh with `graphify update .`.
