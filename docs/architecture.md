# Architecture

Lemon's design centers on one idea: a chatbot that *behaves* like a friend instead of *describing itself* as one. That goal pushes most of the complexity into context preparation rather than generation. The default model is Claude Haiku 4.5 on OpenRouter (`anthropic/claude-haiku-4.5`) for both the main chat and the auxiliary classifier / ToM / state-updater calls — cheap, fast, and supports prompt caching on the persona block.

## High level

```
                    ┌────────────────────────────────────────────────┐
                    │                   config.py                    │
                    │   env, models, paths, knobs, HTTP headers      │
                    └────────────────────────────────────────────────┘
                                          │
        ┌──────────┬───────────┬──────────┼──────────┬───────────┬──────────┐
        ▼          ▼           ▼          ▼          ▼           ▼          ▼
   prompt.py   time_…ctx   history.py  state.py   facts.py    db.py     chat.py
   (persona)   (block)    (compress/   (load,      (block    (SQLite)  (caching,
                          swap)         save,       from               streaming,
                                        update)    facts.db)           pacing)
                                          │
                              ┌───────────┼───────────┐
                              ▼           ▼           ▼
                          emotion.py    tom.py     memory.py     ◀── empathy pipeline
                          (classify)    (ToM)      (retrieve)
                              │           │           │
                              └───────────┼───────────┘
                                          ▼
                                    pipeline.py  ──▶  empathy_check.py
                                    (orchestrator)    (sentiment-mirror)
                                          │
                                          ▼
                                    commands.py  ◀──── chat loop
                                    (slash dispatcher) (lem.py / web.py)
```

Two entry points, one core:

- `lem.py` — terminal REPL.
- `web.py` — FastAPI app + single-page HTML in `templates/index.html`. Exposes the same chat + slash-command + introspection surface over HTTP, plus a `/trace` endpoint for the last empathy-pipeline trace.

## What gets sent to the model each turn

The empathy pipeline (`pipeline.run_empathy_turn`) prepares context, then calls the chat model. The full sequence per user message:

```
user_msg arrives
  │
  ▼
[1] emotion classifier (Haiku)   →  {primary, intensity, undertones, underlying_need}
  ▼                                  ↳ also stored on the message row in db
[2] memory retrieval             →  past user messages with same emotion (other sessions)
  ▼
[3] theory-of-mind pass (Haiku)  →  {feeling, avoid, what_helps}
  ▼
[4] inject system blocks
  ▼
[5] main generation (chat model) →  buffered draft
  ▼
[6] sentiment-mirror check       →  pass? regenerate-once with critique?
  ▼
final reply  →  played back token-by-token with humanize_delay
```

The message list sent to the chat model:

```
0. system: <Who you are>...                        ← persona, ~5KB
1. system: <time_context>...                       ← refreshed each turn
2. system: <internal_state>...                     ← refreshed each turn
3. system: <user_facts>...                         ← refreshed each turn (if any)
4. system: <emotional_memory>...                   ← from pipeline (if memories found)
5. system: <user_emotion>...                       ← from pipeline
6. system: <theory_of_mind>...                     ← from pipeline
7. system: <earlier_conversation>...               ← from compress_history (optional)
... user / assistant / user / ... (last KEEP_RECENT_TURNS)
N. user: <latest message>
```

If `LEMON_PROMPT_CACHE=1` (Anthropic models only) the persona block is wrapped with `cache_control: ephemeral`. With the default OpenAI route, caching is automatic on the prefix — no code-side wrapping needed.

The memory gradient (`history.compress_history`) keeps the most recent `KEEP_RECENT_TURNS` (default 8) verbatim and folds older turns into a single `<earlier_conversation>` summary block.

## State machine

There are three pieces of persistent state in lemon:

1. **Internal state** — a 6-field dict (mood, energy, engagement, emotional_thread, recent_activity, disposition) describing how lemon feels right now. Updated by a small LLM call every `STATE_UPDATE_EVERY` exchanges. Persisted as a snapshot row in `state_snapshots`.

2. **User facts** — a key/value table (`facts`) for things lemon should remember about the user. Populated via `/remember`. Loaded as a `<user_facts>` system block.

3. **Emotion-tagged messages** — every user message gets an `emotion`, `intensity`, `salience` triple from the classifier. Stored on the `messages` row. Used by the memory-retrieval step to surface past moments that felt similar.

## Update cadence

| event                       | what runs                                                                                       |
| --------------------------- | ----------------------------------------------------------------------------------------------- |
| every user message          | empathy pipeline: classify → retrieve → ToM → draft → check                                     |
| every user message          | refresh `<time_context>` + `<internal_state>` + `<user_facts>` system blocks                    |
| every chat call             | buffered generation (so the post-check can run), then humanized replay through stdout/SSE       |
| every `STATE_UPDATE_EVERY`  | call the cheap state-updater model with the latest exchange                                     |
| state change                | save snapshot to `state_snapshots`                                                              |
| every user message          | log row in `messages` with detected emotion fields                                              |
| every assistant reply       | log row in `messages` (no emotion fields)                                                       |
| session end                 | stamp `ended_at` in `sessions`                                                                  |

## Cost per turn

With the default settings:

| call                  | model                              | runs                     | rough cost              |
| --------------------- | ---------------------------------- | ------------------------ | ----------------------- |
| emotion classifier    | `anthropic/claude-haiku-4.5`       | every turn               | ~$0.0001                |
| ToM pass              | `anthropic/claude-haiku-4.5`       | every turn               | ~$0.0002                |
| main chat             | `anthropic/claude-haiku-4.5`       | every turn (+1 on retry) | depends on context size; persona cached |
| state updater         | `anthropic/claude-haiku-4.5`       | every 2 turns            | ~$0.0001                |

The persona system block (~5KB, stable across every turn) is sent with `cache_control: ephemeral` on the main chat call, so after the first turn it's a cache hit. Auxiliary calls (emotion, ToM, state) are short single-shot prompts and don't benefit from caching.

Disable the empathy pipeline entirely with `LEMON_EMPATHY=0` or `/empathy off` to drop back to one chat call per turn.

## Why these design choices

**SQLite, not JSON.** Multiple sessions over time, fact lookup, snapshot history, emotion-tagged retrieval — all relational concerns. SQLite gets these for free with no server.

**Buffer, then replay.** The post-check needs the full draft before it can decide whether to regenerate. Streaming raw tokens to the user would mean shipping a bad draft before the check runs. Buffer the generation, run the check, then replay tokens with humanized pacing. The web UI shows phase events ("reading you...", "thinking...", "rephrasing...") to fill the gap.

**Auxiliary calls all on Haiku.** They produce structured JSON, not user-facing text, so a small fast model is appropriate. The main chat model can stay strong (GPT-5.4-mini) for the actual reply.

**One process, one chat (web).** The web UI assumes the user is running it for themselves on localhost. No auth, single in-memory `ChatContext` guarded by a lock. Multi-user would need session cookies, per-user db rows, and per-user state.

**Pipeline is opt-out, not opt-in.** Default-on so first-time users see the value. `LEMON_EMPATHY=0` (or `/empathy off`) drops back to one chat call per turn with no code change.

**Pure functions where possible.** `parse_state_response`, `format_internal_state`, `compress_history`, `replace_system_block`, `humanize_delay`, `time_of_day_label`, `session_duration_note`, `check_response`, `format_emotion_block`, `format_tom_block`, `format_memory_block` — all take inputs and return outputs with no side effects. Trivially testable, and they're what most of the test suite covers.

## Adding a feature

The pattern: most new features land as a new module in `src/` plus a test file in `tests/`. The `commands.py` registry exposes anything user-controllable as a slash command without touching the loop. The web UI inherits new commands automatically — no client-side changes needed.

For empathy-pipeline extensions specifically (e.g. best-of-N, RAG, multi-agent critic — see `docs/empathy_research.md` Tier 2/3), edit `pipeline.run_empathy_turn` and add the step in the right spot. The trace dataclass auto-surfaces in `/why` and `/trace` for any new field you add to it.
