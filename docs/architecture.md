# Architecture

Lemon's design centers on one idea: a chatbot that *behaves* like a friend instead of *describing itself* as one. That goal pushes most of the complexity into context preparation rather than generation. The default model is Claude Haiku 4.5 on OpenRouter (`anthropic/claude-haiku-4.5`) for both the main reply and the merged pre-/post-generation classifier calls — cheap, fast, and supports prompt caching on the persona block.

## High level

```
                       ┌──────────────────────────────────────────┐
                       │                 config.py                │
                       │     env, models, paths, knobs, HTTP      │
                       └──────────────────────────────────────────┘
                                           │
   ┌─────────────┬───────────────┬─────────┼─────────┬───────────────┬──────────────┐
   ▼             ▼               ▼         ▼         ▼               ▼              ▼
prompt/       empathy/          llm/              storage/       session_        pipeline.py
(persona,     (emotion,        (chat,             (db,           context.py     (orchestrator:
 time,         tom, check,     parse_utils)       memory,        (initial       read_user →
 history,      user_read,                         state)         history,       memory → draft
 facts)        post_exchange,                                    refresh,       → check → regen)
               fact_extractor)                                   bookkeeping)
                                                                      │
                                                                      ▼
                                                              lem.py / web.py
                                                           + commands.py (slash)
```

Two entry points, one core:

- `lem.py` — terminal REPL.
- `web.py` — FastAPI app + single-page HTML in `templates/index.html`. Same chat + slash-command + introspection surface over HTTP, plus a `/trace` endpoint.

## What gets sent to the model each turn

`pipeline.run_empathy_turn` prepares context, then calls the chat model. Full sequence per user message:

```
user_msg arrives
  │
  ▼
[1] merged pre-gen read (Haiku)   →  emotion {primary, intensity, undertones, underlying_need}
    empathy.user_read.read_user      tom     {feeling, avoid, what_helps}
  │                                  ↳ one LLM call, two output dicts
  │                                  ↳ emotion also stored on the message row in db
  ▼
[2] memory retrieval              →  past user messages with same emotion (other sessions)
    storage.memory.relevant_memories   — SQLite only, no LLM
  ▼
[3] inject system blocks
  ▼
[4] main generation (chat model)  →  buffered draft via llm.chat.generate_reply
  ▼
[5] sentiment-mirror check        →  12 regex detectors; pass? regenerate-once with critique?
  ▼
final reply ships to user (SSE "token" + "done", or CLI print)
  │
  ▼
[6] backgrounded bookkeeping (Haiku)  →  new_facts + nudged_state, merged into one call
    empathy.post_exchange.bookkeep      — runs in a daemon thread AFTER the reply is delivered
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

If `LEMON_PROMPT_CACHE=1` (Anthropic models only) the persona block is wrapped with `cache_control: ephemeral`.

The memory gradient (`prompt.history.compress_history`) keeps the most recent `KEEP_RECENT_TURNS` (default 8) verbatim and folds older turns into a single `<earlier_conversation>` summary block.

## State machine

Three pieces of persistent state:

1. **Internal state** — a 6-field dict (mood, energy, engagement, emotional_thread, recent_activity, disposition). Nudged every turn by the merged post-gen `bookkeep` call. Persisted as a snapshot row in `state_snapshots`.
2. **User facts** — a key/value table (`facts`) for things lemon should remember across sessions. Populated by `/remember` and by `bookkeep`. Loaded as a `<user_facts>` system block each turn.
3. **Emotion-tagged messages** — every user message gets an `emotion`, `intensity`, `salience` triple from the pre-gen read. Stored on the `messages` row. Used by memory-retrieval to surface past moments that felt similar.

## Update cadence

| event                       | what runs                                                                       |
| --------------------------- | ------------------------------------------------------------------------------- |
| every user message          | empathy pipeline: user_read → retrieve → draft → check → optional retry         |
| every user message          | refresh `<time_context>` + `<internal_state>` + `<user_facts>` system blocks    |
| every user message          | log row in `messages` with detected emotion fields                              |
| every assistant reply       | log row in `messages` (no emotion fields)                                       |
| every user message          | daemon thread: `post_exchange.bookkeep` → upsert facts + save new state snapshot |
| session end                 | stamp `ended_at` in `sessions`                                                  |

## Cost per turn

| call                         | model                         | runs                     | rough cost |
| ---------------------------- | ----------------------------- | ------------------------ | ---------- |
| `user_read` (emotion + ToM)  | `anthropic/claude-haiku-4.5`  | every turn               | ~$0.0002   |
| main chat                    | `anthropic/claude-haiku-4.5`  | every turn (+1 on retry) | varies; persona cached after turn 1 |
| `bookkeep` (facts + state)   | `anthropic/claude-haiku-4.5`  | every turn, backgrounded | ~$0.0002   |

**User-perceived latency = 2 LLM calls.** Total cost = 3 per typical turn.

Disable the empathy pipeline with `LEMON_EMPATHY=0` or `/empathy off` to drop to one user-facing chat call per turn (bookkeep still runs in the background).

## Why these design choices

**SQLite, not JSON.** Multiple sessions over time, fact lookup, snapshot history, emotion-tagged retrieval — all relational concerns.

**Buffer, then replay.** The post-check needs the full draft before deciding whether to regenerate. Buffer generation, run the check, then ship the final reply as a single SSE payload. Phase events fill the wait visually.

**Merged classifiers.** The pre-generation read and the post-generation bookkeeping each used to be two separate LLM calls. Now they're one call apiece — both the pre-gen pair and the post-gen pair were hitting the same model with overlapping inputs. Halves the round-trips with negligible quality cost.

**Bookkeep in a daemon thread.** Post-reply work (fact extraction + state nudge) doesn't need to block the user. Fires after `done` is delivered, so user-perceived latency drops by whatever that call costs (~1-2s typical).

**Auxiliary calls all on Haiku.** They produce structured JSON, not user-facing text, so a small fast model is appropriate.

**One process, one chat (web).** The web UI assumes the user is running it for themselves on localhost. No auth, single in-memory `ChatContext` guarded by a lock.

**Pipeline is opt-out, not opt-in.** Default-on so first-time users see the value.

**Pure functions where possible.** Parsers, validators, formatters, compressors, regex detectors — all side-effect-free. Trivially testable and that's what most of the test suite covers.

## Adding a feature

Most new features land as a new module in the appropriate package (`prompt/`, `empathy/`, `llm/`, or `storage/`) plus a test file in `tests/`. The `commands.py` registry exposes anything user-controllable as a slash command without touching the loop — the web UI inherits new commands automatically.

For empathy-pipeline extensions specifically (e.g. best-of-N, RAG, multi-agent critic — see `docs/empathy_research.md` Tier 2/3), edit `pipeline.run_empathy_turn` and add the step in the right spot. The trace dataclass auto-surfaces in `/why` and `/trace` for any new field you add.
