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
prompts.py    empathy/          llm/              storage/       session_        pipeline.py
persona.py    (emotion,        (chat,             (db, memory,   context.py     (orchestrator:
              tom, check,      parse_utils)       lemon_state,   (initial       read_user →
              user_read,                          user_state,    history,       memory → draft
              post_exchange,                      state-shim)    refresh,       → check → regen)
              fact_extractor)                                    bookkeeping)
                                                                      │
                                                                      ▼
                                                              lem.py / web.py
                                                           + commands.py (slash)
```

`persona.py` exports `LEMON_TRAITS` (Big 5) and `LEMON_ADAPTATIONS` (goals
/ values / concerns / stance) — the static seeds of lemon's three-layer
state. `storage/lemon_state.py` and `storage/user_state.py` hold the
runtime three-layer schema for both agents. The legacy `storage/state.py`
remains as a deprecated shim for the legacy migration path. See
`docs/dyadic_state.md` for the full schema and rationale.

Two entry points, one core:

- `lem.py` — terminal REPL.
- `web.py` — FastAPI app + single-page HTML in `templates/index.html`. Same chat + slash-command + introspection surface over HTTP, plus a `/trace` endpoint.

## What gets sent to the model each turn

`pipeline.run_empathy_turn` prepares context, then calls the chat model. Full sequence per user message:

```
user_msg arrives
  │
  ▼
[1] merged pre-gen read (Haiku)   →  emotion           {primary, intensity, undertones, underlying_need}
    empathy.user_read.read_user      tom               {feeling, avoid, what_helps}
                                     user_state_delta  PAD nudge + adaptation churn for the user
                                     lemon_state_delta PAD nudge + adaptation churn for lemon (damped)
  │                                  ↳ one LLM call, four output dicts
  │                                  ↳ emotion also stored on the message row in db
  │                                  ↳ both deltas applied + both states persisted before reply
  ▼
[2] memory retrieval              →  past user messages, composite-scored
    storage.memory.relevant_memories   — SQLite only, no LLM
  ▼
[3] inject system blocks (lemon_state, user_state, reading, memory)
  ▼
[4] main generation (chat model)  →  buffered draft via llm.chat.generate_reply
                                     reads from BOTH freshly-nudged states
  ▼
[5] sentiment-mirror check        →  12 regex detectors; pass? regenerate-once with critique?
  ▼
final reply ships to user (SSE "token" + "done", or CLI print)
  │
  ▼
[6] backgrounded bookkeeping (Haiku)  →  new_facts only (state nudges happened pre-reply)
    empathy.post_exchange.bookkeep       — runs in a daemon thread AFTER the reply is delivered
```

The message list sent to the chat model:

```
0. system: <Who you are>...                  ← persona, ~5KB, prompt-cached
1. system: <time_context>...                 ← refreshed each turn
2. system: <lemon_state>...                  ← refreshed each turn, then re-overwritten with the post-nudge state
3. system: <user_facts>...                   ← refreshed each turn (if any)
4. system: <emotional_memory>...             ← from pipeline (if memories found)
5. system: <user_state>...                   ← from pipeline (user's tonic state, three-layer)
6. system: <reading>...                      ← from pipeline (unified phasic emotion + ToM)
7. system: <earlier_conversation>...         ← from compress_history (optional)
... user / assistant / user / ... (last KEEP_RECENT_TURNS)
N. user: <latest message>
```

Tonic-then-phasic ordering for both agents: `<lemon_state>` (lemon's tonic)
sits at the top; `<user_state>` (user's tonic) and `<reading>` (per-turn
phasic emotion + ToM) sit per-turn at the bottom. Stage 3 of the dyadic-state
architecture folded the old `<user_emotion>` and `<theory_of_mind>` blocks
into `<reading>`.

If `LEMON_PROMPT_CACHE=1` (Anthropic models only) the persona block is wrapped with `cache_control: ephemeral`.

The memory gradient (`prompt_stack.compress_history`) keeps the most recent `KEEP_RECENT_TURNS` (default 8) verbatim and folds older turns into a single `<earlier_conversation>` summary block.

## State machine

Four pieces of persistent state, two of them three-layer dyadic objects:

1. **Lemon state** — a three-layer dict (Big 5 traits + characteristic adaptations + PAD core affect with a derived mood label). Traits hardcoded in `persona.LEMON_TRAITS`; adaptations seeded from `persona.LEMON_ADAPTATIONS`; PAD nudged every turn pre-reply by the merged user_read pass (with asymmetric damping so lemon stays stable). Persisted as a snapshot row in `lemon_state_snapshots`.
2. **User state** — same shape as lemon. Traits inferred slowly (per-turn nudge cap effectively freezes them in stage 1); adaptations grow as the user shares goals, values, and concerns; PAD updates per turn. Persisted as a snapshot row in `user_state_snapshots`.
3. **User facts** — a key/value table (`facts`) for things lemon should remember across sessions. Populated by `/remember` and by `bookkeep` (post-reply, facts-only since stage 2). Loaded as a `<user_facts>` system block each turn.
4. **Emotion-tagged messages** — every user message gets an `emotion`, `intensity`, `salience` triple from the pre-gen read. Stored on the `messages` row. Used by memory-retrieval to surface past moments that felt similar.

The legacy `state_snapshots` table from the pre-stage-3 6-field schema stays as archive; new writes go to the two new tables.

## Update cadence

| event                       | what runs                                                                                          |
| --------------------------- | -------------------------------------------------------------------------------------------------- |
| every user message          | empathy pipeline: user_read → retrieve → draft → check → optional retry                            |
| every user message          | refresh `<time_context>` + `<lemon_state>` + `<user_facts>` system blocks                          |
| every user message          | merged user_read pass → emotion + tom + user_state_delta + lemon_state_delta in one LLM call       |
| every user message          | apply both deltas, persist both via `lemon_state_snapshots` / `user_state_snapshots`               |
| every user message          | log row in `messages` with detected emotion fields                                                 |
| every assistant reply       | log row in `messages` (no emotion fields)                                                          |
| every user message          | daemon thread: `post_exchange.bookkeep` → facts only (state already updated pre-reply)             |
| session end                 | stamp `ended_at` in `sessions`                                                                     |

## Cost per turn

| call                                                       | model                         | runs                     | rough cost |
| ---------------------------------------------------------- | ----------------------------- | ------------------------ | ---------- |
| `user_read` (emotion + ToM + user_delta + lemon_delta)     | `anthropic/claude-haiku-4.5`  | every turn               | ~$0.0002–0.0003 |
| main chat                                                  | `anthropic/claude-haiku-4.5`  | every turn (+1 on retry) | varies; persona cached after turn 1 |
| `bookkeep` (facts only)                                    | `anthropic/claude-haiku-4.5`  | every turn, backgrounded | ~$0.0001   |

**User-perceived latency = 2 LLM calls.** Total cost = 3 per typical turn.

Stage 2 of the dyadic state architecture folded the lemon-side state nudge into the existing user_read call instead of adding a new round trip. The user_read JSON output is longer (4 sub-objects instead of 2) but the call count is unchanged. `bookkeep` got *cheaper* in the same change because its state half is gone.

Disable the empathy pipeline with `LEMON_EMPATHY=0` or `/empathy off` to drop to one user-facing chat call per turn (bookkeep still runs in the background; with the pipeline off, no state nudges happen at all that turn).

## Why these design choices

**SQLite, not JSON.** Multiple sessions over time, fact lookup, snapshot history, emotion-tagged retrieval — all relational concerns.

**Buffer, then replay.** The post-check needs the full draft before deciding whether to regenerate. Buffer generation, run the check, then ship the final reply as a single SSE payload. Phase events fill the wait visually.

**Merged classifiers.** The pre-generation read used to be two separate calls (emotion + ToM); now it's one call with four sub-objects (emotion + ToM + user_state_delta + lemon_state_delta). Same round-trip count as before, just denser output. The post-generation bookkeeping shrunk from facts+state to facts-only because state moved pre-reply.

**State first, response second.** Stage 2 of the dyadic state architecture moved both agents' state nudges to *before* the reply (inside the merged user_read pass). The reply call reads from a freshly-updated `<lemon_state>` and `<user_state>`, so reply tone genuinely reflects what lemon "feels" in response to what was just said — instead of the older flow where state caught up after the fact.

**Symmetric schema, asymmetric dynamics.** Both agents use the same three-layer state object, but lemon's PAD nudges are clamped tighter (±0.10 vs ±0.15), her traits and values are persona-fixed and never drift, and her relational stance re-pegs on session start. The user's state has no session-start overrides because realistically a user brings their carried-in mood into the next conversation.

**Bookkeep in a daemon thread.** Post-reply fact extraction doesn't need to block the user. Fires after `done` is delivered, so user-perceived latency drops by whatever that call costs (~1-2s typical).

**Auxiliary calls all on Haiku.** They produce structured JSON, not user-facing text, so a small fast model is appropriate.

**One process, one chat (web).** The web UI assumes the user is running it for themselves on localhost. No auth, single in-memory `ChatContext` guarded by a lock.

**Pipeline is opt-out, not opt-in.** Default-on so first-time users see the value.

**Pure functions where possible.** Parsers, validators, formatters, compressors, regex detectors — all side-effect-free. Trivially testable and that's what most of the test suite covers.

## Adding a feature

Most new features land as a new module in the appropriate package (`prompt/`, `empathy/`, `llm/`, or `storage/`) plus a test file in `tests/`. The `commands.py` registry exposes anything user-controllable as a slash command without touching the loop — the web UI inherits new commands automatically.

For empathy-pipeline extensions specifically (e.g. best-of-N, RAG, multi-agent critic — see `docs/empathy_research.md` Tier 2/3), edit `pipeline.run_empathy_turn` and add the step in the right spot. The trace dataclass auto-surfaces in `/why` and `/trace` for any new field you add.
