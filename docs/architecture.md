# Architecture

Lemon's design centers on one idea: a chatbot that *behaves* like a friend instead of *describing itself* as one. That goal pushes most of the complexity into context preparation rather than generation. The model itself is mostly off-the-shelf Claude on OpenRouter.

## High level

```
                  ┌─────────────────────────────────────────────┐
                  │                  config.py                  │
                  │   env, models, paths, knobs, HTTP headers   │
                  └─────────────────────────────────────────────┘
                                       │
        ┌──────────┬─────────┬─────────┼─────────┬───────────┬──────────┐
        ▼          ▼         ▼         ▼         ▼           ▼          ▼
   prompt.py   time_…    history.py  state.py  facts.py   db.py     chat.py
   (persona)   ctx     (compress/   (load,     (block    (SQLite)  (caching,
                       swap)         save,      from              streaming,
                                     update)   facts.db)          pacing)
        │          │         │         │         │           │          │
        └──────────┴─────────┴─────────┴─────────┴───────────┘          │
                                       │                                │
                                       ▼                                ▼
                                 commands.py ◀───────────────────── chat loop
                                 (slash dispatcher)                 (lem.py / web.py)
```

Two entry points, one core:

- `lem.py` — terminal REPL.
- `web.py` — FastAPI app + single-page HTML in `templates/index.html`. Exposes the same chat + slash-command + introspection surface over HTTP.

## What gets sent to the model each turn

Every chat call sends a list of messages roughly shaped like this:

```
1. system: <Who you are>...                        ← persona, ~5KB, CACHED
2. system: <time_context>...                       ← regenerated each turn
3. system: <internal_state>...                     ← regenerated each turn
4. system: <user_facts>...                         ← regenerated each turn (if any)
5. system: <earlier_conversation>...               ← only if older turns exist
6. user / assistant / user / ... (last 8 turns)    ← memory gradient
N. user: <latest message>
```

Position 1 (the persona) is the *only* block marked with Anthropic-style `cache_control: ephemeral`. It's stable across every turn, so the cache hit pays for itself within two messages. Positions 2–4 change every turn and stay uncached.

The memory gradient (`history.compress_history`) keeps the most recent `KEEP_RECENT_TURNS` (default 8) verbatim and folds older turns into a single `<earlier_conversation>` summary block. This bounds context cost without losing topic memory.

## State machine

There are two pieces of state in lemon:

1. **Internal state** — a 6-field dict (mood, energy, engagement, emotional_thread, recent_activity, disposition). Persisted as a snapshot row in `state_snapshots` after every change. Loaded on startup as the most recent row.

2. **User facts** — a key/value table (`facts`) used as long-term memory about the user. Currently populated only via the `/remember` slash command. A future extractor agent could populate this automatically after each session.

## Update cadence

| event                       | what runs                                                          |
| --------------------------- | ------------------------------------------------------------------ |
| every user message          | refresh `<time_context>` + `<internal_state>` + `<user_facts>`     |
| every user message          | recompute compression (no-op until the threshold)                  |
| every chat call             | stream tokens, print or relay via SSE, sleep `humanize_delay()`    |
| every `STATE_UPDATE_EVERY`  | call the cheap state-updater model with the latest exchange        |
| state change                | save snapshot to `state_snapshots`                                 |
| every message (user/asst)   | log row in `messages` table                                        |
| session end                 | stamp `ended_at` in `sessions` table                               |

## Why these design choices

**SQLite, not JSON.** Multiple sessions over time, fact lookup, snapshot history — all relational concerns. SQLite gets these for free with no server. The original `.lemon_state.json` was a single overwritten blob and lost everything on every save.

**One process, one chat (web).** The web UI assumes the user is running it for themselves on localhost. There is no auth, no multi-tenant separation, and a single in-memory `ChatContext` guarded by a lock. That's a deliberate scope cap; multi-user would need session cookies, per-user db rows, and per-user state.

**Cacheable persona, not whole prompt.** Cache breakpoints have a cost: the prefix has to match exactly. Time and state blocks change every turn, so caching them never hits. The persona is ~5KB of stable text and pays back immediately.

**Humanize in the streaming loop.** The model returns tokens as fast as possible. We add per-token sleep scaled by the `energy` field so a tired lemon types slower than an upbeat one. This is the only place where internal state visibly affects the *delivery* of output, not just its content.

**Pure functions where possible.** `parse_state_response`, `format_internal_state`, `compress_history`, `replace_system_block`, `humanize_delay`, `time_of_day_label`, `session_duration_note` — all take inputs and return outputs with no side effects, which makes them trivially testable.

## Adding a feature

The pattern: most new features land as a new module in `src/` plus a test file in `tests/`. The `commands.py` registry exposes anything user-controllable as a slash command without touching the loop. The web UI inherits new commands automatically — no client-side changes needed.
