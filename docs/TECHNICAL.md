# Technical Reference

A single-document walkthrough of lemon's implementation: every module, every call path, every piece of state, every request shape.

This complements rather than replaces the existing docs. For narrative context see `architecture.md`. For clinical empathy theory see `empathy_research.md`. For schema details see `db_schema.md`. This file is the "how it actually runs" reference.

---

## 1. Overview

Lemon is a chat assistant styled as a friend, not a productivity tool. Implementation is a small Python codebase (~1.5k LoC in `src/`) with two frontends over one backend:

- **Backend core:** per-turn "empathy pipeline" that runs one merged pre-gen LLM call + one main reply call (+ optional retry) + one merged post-gen LLM call.
- **CLI:** `src/lem.py`, a stdin/stdout REPL.
- **Web:** `src/web.py`, FastAPI + a single hand-written HTML page + Server-Sent Events for streaming.
- **Persistence:** one SQLite file with four tables (sessions, messages, state_snapshots, facts). Idempotent schema + migration table.
- **Model layer:** OpenRouter as the HTTP target, Anthropic Claude Haiku 4.5 as the default for both main chat and auxiliary calls. Anthropic-style `cache_control` breakpoints on the persona block when the model supports it.

Everything is synchronous on the critical path; post-gen bookkeeping runs in a daemon thread after the reply is delivered. One process, one conversation at a time. No auth, no multi-user.

**Per-turn LLM budget:**

| call | role | blocks the user? |
|---|---|---|
| `user_read` (STATE_MODEL) | merged emotion + theory-of-mind | yes |
| `generate_reply` (CHAT_MODEL) | the actual reply, streamed | yes |
| `generate_reply` (retry, conditional) | regenerate on empathy-check failure | yes, rare |
| `bookkeep` (STATE_MODEL) | merged fact extraction + state nudge | **no** — backgrounded |

**User-perceived wait = 2 LLM calls.** Total cost = 3 per typical turn (4 when retry fires).

---

## 2. Setup, configuration, environment

### 2.1 Install and run

```bash
pip install -r requirements.txt
echo 'OPENROUTER_API_KEY=sk-or-v1-...' > .env
python src/lem.py            # CLI
python src/web.py            # web UI on 127.0.0.1:8000
pytest                       # test suite (after pip install -r requirements-dev.txt)
```

### 2.2 Environment variables

All parsed in `src/config.py` at import time. Override any of them via `.env` or shell export.

| variable | default | meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | (required) | raises `ValueError` at import if missing |
| `LEMON_CHAT_MODEL` | `anthropic/claude-haiku-4.5` | main generation model |
| `LEMON_STATE_MODEL` | `anthropic/claude-haiku-4.5` | pre-gen read + post-gen bookkeeping |
| `LEMON_PROMPT_CACHE` | auto (`1` for `anthropic/*`, else `0`) | wrap persona block in `cache_control: ephemeral` |
| `LEMON_EMPATHY` | `1` | master switch for the pipeline |
| `LEMON_EMPATHY_RETRY` | `1` | regenerate once when the post-check fails |
| `LEMON_MEMORY_LIMIT` | `3` | how many matching-emotion memories to inject |
| `LEMON_AUTO_FACTS` | `1` | enable post-exchange fact extraction + state nudge |
| `LEMON_AUTO_FACTS_MAX` | `3` | max facts stored per turn |
| `LEMON_DB` | `.lemon.db` | SQLite file path (gitignored) |

### 2.3 In-code constants (not env-settable)

From `src/config.py`:

```python
KEEP_RECENT_TURNS = 8           # before compress_history folds older turns
```

### 2.4 HTTP wiring

```python
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost/lemon",
    "X-Title": "lemon chat",
}
```

Three modules call this endpoint: `llm/chat.py`, `empathy/user_read.py`, `empathy/post_exchange.py`.

---

## 3. Module map

```
src/
  config.py              env, models, paths, knobs, HTTP headers
  pipeline.py            orchestrator: the "empathy pipeline"
  session_context.py     shared CLI+web helpers: initial history, refresh blocks, bookkeeping thread
  commands.py            slash-command registry + ChatContext + dispatcher
  lem.py                 CLI REPL entry point
  web.py                 FastAPI app + SSE + introspection endpoints

  prompt/                system-prompt content lemon reads
    persona.py           persona system-prompt string + opener pool
    time_context.py      <time_context> block generator
    history.py           replace_system_block + compress_history
    facts.py             <user_facts> block formatter

  empathy/               empathy-pipeline-specific logic
    emotion.py           emotion schema, validator, <user_emotion> formatter
    tom.py               theory-of-mind schema, validator, <theory_of_mind> formatter
    fact_extractor.py    fact-key regex + value-hygiene validator
    empathy_check.py     12-detector regex post-check
    user_read.py         merged pre-gen LLM call (emotion + ToM)
    post_exchange.py     merged post-gen LLM call (facts + state nudge)

  llm/                   raw LLM wire + parsing helpers
    chat.py              OpenRouter reply call, cache wrap, streaming
    parse_utils.py       shared fence-stripper + recent-msgs prompt formatter

  storage/               persistence + retrieval
    db.py                SQLite layer: schema, migrations, CRUD helpers
    memory.py            emotion-tagged message retrieval + <emotional_memory> formatter
    state.py             internal state: defaults, load/save/format + validator

  templates/
    index.html           single-page web UI (vanilla JS, inline CSS)
tests/                   one test file per src/ module + conftest.py
docs/                    architecture, slash commands, db schema, web ui, empathy research
```

Dependency direction flows top-to-bottom:

```
config ──► everything
storage/db ──► storage/state, storage/memory, commands, pipeline, session_context
llm/parse_utils ──► empathy/{emotion,tom,fact_extractor}, storage/state, empathy/{user_read,post_exchange}
llm/chat ──► pipeline
empathy/* ──► pipeline, session_context
prompt/* ──► session_context
commands + session_context + pipeline ──► lem.py, web.py
```

There are no cycles. Tests are isolated via an autouse `isolated_db()` fixture in `tests/conftest.py`.

---

## 4. Per-turn request lifecycle

Full path a single user message takes. Numbered callouts are the LLM calls.

```
user types a message
  │
  ▼
[refresh base blocks]  session_context.refresh_base_blocks()
  replaces <time_context>, <internal_state>, <user_facts> at positions
  1/2/3 via prompt.history.replace_system_block.
  │
  ▼
if is_command(user_input):
    commands.dispatch(user_input, ctx) ──► system bubble; no LLM call; return
  │
  ▼
pipeline.run_empathy_turn(user_msg, base_history, ...)
  │
  ├── if ENABLE_EMPATHY_PIPELINE is False:
  │     append user_msg + compress_history + (R) generate_reply + return
  │
  ├── recent = recent_messages_for_context(base_history, n=6)
  │
  ├── (1) emotion, tom = empathy.user_read.read_user(user_msg, recent)   [STATE_MODEL]
  │       one call, returns both dicts
  │       phase: "reading you"
  │       db.log_message(user_msg, emotion=..., intensity=..., salience=intensity)
  │
  ├── memories = storage.memory.relevant_memories(emotion.primary,       [SQLite only]
  │                  current_session_id, limit=MEMORY_RETRIEVAL_LIMIT)
  │       phase: "remembering"
  │       skipped when primary == "neutral"
  │
  ├── history = base_history, then:
  │     - _inject_block(<emotional_memory>, memories)   if memories
  │     - _inject_block(<user_emotion>, emotion_block)
  │     - _inject_block(<theory_of_mind>, tom_block)
  │
  ├── history.append({role: user, content: user_msg})
  ├── history = compress_history(history, keep_recent=8)
  │
  ├── (R) draft = llm.chat.generate_reply(history, model)                 [CHAT_MODEL]
  │       phase: "replying"
  │       buffered — post-check needs the full reply
  │
  ├── check = empathy.empathy_check.check_response(user_msg, draft, emotion)
  │       12 regex detectors. Pure Python, no LLM call.
  │
  ├── final = draft
  │   if not check.passed and EMPATHY_RETRY_ON_FAIL:
  │       phase: "rephrasing"
  │       retry_history = _inject_block(<empathy_retry>, critique_block)
  │       (R') second = llm.chat.generate_reply(retry_history)            [CHAT_MODEL]
  │       if second.strip(): final = second; trace.regenerated = True
  │
  └── db.log_message(assistant, final); return (final, trace)
  │
  ▼
caller (web._stream_reply or lem.main):
  - ctx.last_trace = trace
  - append user_msg + final to ctx.history
  - deliver the reply (SSE token+done, or CLI print)
  - spawn daemon thread:
      (2) new_facts, new_state = empathy.post_exchange.bookkeep(            [STATE_MODEL]
             user_msg, final, existing_facts, current_state, recent, model)
          db.upsert_fact(...) for each new_fact
          ctx.internal_state = new_state
          db.save_state_snapshot(...)
```

**Total per turn:** 2 LLM calls on the user-facing critical path (user_read + reply), +1 in the background (post-gen bookkeeping), +1 if the post-check fails (retry). With `/empathy off`: 1 main call on the critical path + 1 backgrounded bookkeep.

---

## 5. What the model sees each turn

The `messages` list passed to OpenRouter for the main chat call:

```
 0. system: <Who you are>...                   persona, ~5KB, wrapped in cache_control
 1. system: <time_context>...                  refreshed each turn
 2. system: <internal_state>...                refreshed each turn
 3. system: <user_facts>...                    refreshed each turn, skipped if empty
 4. system: <emotional_memory>...              pipeline-injected, skipped if empty
 5. system: <user_emotion>...                  pipeline-injected
 6. system: <theory_of_mind>...                pipeline-injected
 7. system: <earlier_conversation>...          only when len(convo) > KEEP_RECENT_TURNS
 ...
    user / assistant / user / ...              last KEEP_RECENT_TURNS turns verbatim
 N. user: <latest message>                     appended by pipeline just before generate
```

Key invariants:

- **Order of injection is deterministic.** `_inject_block` drops any prior block with the same tag, then inserts the new content just after the leading contiguous block of system messages.
- **`time_context` + `internal_state` + `user_facts` change every turn** and by construction live at positions 1, 2, 3 (managed by `replace_system_block` in `prompt.history`).
- **Persona block never changes.** That is what makes the cache hit work.
- **Emotion/ToM/Memory blocks are pipeline-scoped.** They live inside the history temporarily for one call, then are dropped when the pipeline rebuilds next turn.

### 5.1 Persona block (`prompt.persona.LEMON_PROMPT`)

Sections inside `<Who you are>` through `<forbidden words>`. Notable:

- `<Who you are>`: role definition (friend, not assistant, no gender).
- `<Voice and tone rules>`: WhatsApp-register constraints.
- `<internal_state_instructions>`: how to read the state block without narrating it.
- `<time_aware_personality>`: tone rules per time-of-day and session length.
- `<rules_of_time>`: irreversibility, causality, duration-asymmetry axioms.
- `<language mirroring>`: Hinglish-default, match user's language.
- `<conversation rules>`: "only respond to what the user said", no unsolicited advice, short by default.
- `<formatting>`: no hyphens, no lists, keep messages flowing.
- `<forbidden words>`: negative-list ("Vibe", "quiet", "great to see you", "What is on your mind").

The block is ~5KB. It never changes between turns, so Anthropic-style prompt caching is effectively free after the first call.

### 5.2 Refreshed-per-turn blocks

#### `<time_context>` (`prompt.time_context.get_time_context`)

```
<time_context>
Current local date: 2026-04-23
Current local time: 15:42
Day of week: Thursday
Time of day: afternoon
You've been talking for a bit now, around 18 minutes.
</time_context>
```

Time buckets: morning (5-9), afternoon (10-16), evening (17-20), late night (21-23), very late night (0-4).

#### `<internal_state>` (`storage.state.format_internal_state`)

Rendered from the 6-field dict. Defaults in `storage.state.DEFAULT_STATE`:

```python
{"mood": "neutral", "energy": "medium", "engagement": "normal",
 "emotional_thread": None, "recent_activity": None, "disposition": "warm"}
```

#### `<user_facts>` (`prompt.facts.format_user_facts`)

Skipped entirely when the `facts` table is empty.

### 5.3 Pipeline-injected blocks

#### `<user_emotion>` (`empathy.emotion.format_emotion_block`)

```
<user_emotion>
Primary feeling: sadness (moderate, intensity 0.52)
Undertones: loneliness, tired
What they probably want: feel heard, not solved
</user_emotion>
```

Intensity word ladder: `<0.3 mild`, `<0.6 moderate`, `<0.85 strong`, else `very strong`.

#### `<theory_of_mind>` (`empathy.tom.format_tom_block`)

```
<theory_of_mind>
What they're feeling: tired and a little embarrassed about the exam
Don't: don't jump to advice
Do: stay with it, ask one open question if anything
</theory_of_mind>
```

#### `<emotional_memory>` (`storage.memory.format_memory_block`)

```
<emotional_memory>
- yesterday, when feeling sadness: "i just feel flat about this..."
- 4 days ago, when feeling sadness: "everyone's moved on and I'm still stuck"
</emotional_memory>
```

Timestamps are humanized via `storage.memory._humanize_age`: `today`, `yesterday`, `N days ago`, `N weeks ago`, `N months ago`.

---

## 6. Merged pre-gen call — `empathy.user_read.read_user`

One LLM round-trip that returns `(emotion_dict, tom_dict)` — replacing two separate calls that used to fire sequentially.

- Model: `STATE_MODEL`. `temperature=0.3`, `max_tokens=500`.
- Input: the user message + last ~6 non-system turns.
- Output: a single JSON object with `"emotion"` and `"tom"` sub-dicts.
- Parsing: `llm.parse_utils.strip_json_fences` → `json.loads` → split sub-dicts → `empathy.emotion._validate(dict)` + `empathy.tom._validate(dict)` (no re-dumping).
- Fallback: `(DEFAULT_EMOTION, DEFAULT_TOM)` on any failure. Chat never dies because this call choked.
- Side effect: the user message row in `messages` is written with `emotion`, `intensity`, and `salience=intensity` populated by the pipeline after this call returns.

The two module-level validators (`_validate` in both `empathy/emotion.py` and `empathy/tom.py`) enforce the label whitelist, intensity clamp, short-string coercion, etc. Each `_parse(raw)` in those modules is now a thin wrapper over `_validate(json.loads(strip_json_fences(raw)))`, kept in case anything wants to parse a raw string directly.

---

## 7. Main chat call — `llm/chat.py`

### 7.1 Shape

```python
payload = {
    "model":             model or CHAT_MODEL,
    "temperature":       0.75,
    "top_p":             0.95,
    "frequency_penalty": 0.2,
    "max_tokens":        400,
    "messages":          prepare_messages(history),
    "stream":            True,
}
```

`frequency_penalty=0.2` suppresses "I hear you, that's so valid" style phrase-stacking.

### 7.2 Prompt caching (`prepare_messages`)

Only runs if `ENABLE_PROMPT_CACHE` is true. It wraps the persona block into Anthropic's structured-content form:

```python
{"role": "system",
 "content": [{"type": "text", "text": content,
              "cache_control": {"type": "ephemeral"}}]}
```

Any system message containing `<Who you are>` (matched by `PERSONA_TAG`) is wrapped. Everything else stays as a plain string, so other system blocks go uncached. After the first call, the persona is a cache hit on the Anthropic side — subsequent turns pay roughly zero cost for those 5KB.

### 7.3 Streaming

Two entry points:
- `iter_chat(history, model)` — generator yielding content deltas from the SSE upstream.
- `generate_reply(history, model)` — buffered, concatenates `iter_chat` output. Used by the empathy pipeline so the post-check can run on a complete string.

The web UI buffers via `generate_reply`, then delivers the full reply in one SSE `token` event followed by `done`.

---

## 8. Empathy post-check — `empathy/empathy_check.py`

Pure Python, regex-based, zero LLM cost. Runs after the main draft. **12 detectors** as of the robustness pass:

| detector | fires when |
|---|---|
| `minimizing` | substring match on `MINIMIZING` OR sentence-start "at least" |
| `toxic_positivity` | cliché phrases ("silver lining", "on the bright side", "the good news is", etc.) |
| `advice_pivot` | draft *opens* with advice pattern AND user emotion is negative, intensity ≥ 0.5 |
| `polarity_mismatch` | draft *opens* with cheery opener AND user emotion is negative, intensity ≥ 0.4 |
| `validation_cascade` | 3+ validation phrases anywhere, OR 2+ in the first 80 chars |
| `therapy_speak` | clinical labeling ("sounds like anxiety", "textbook trauma", "you're catastrophizing") |
| `self_centering` | distress-gated opener centering responder's reaction ("i wish i could fix this for you") |
| `sycophancy` | agreement inflation ("great question", "you're so right", "couldn't agree more") |
| `false_equivalence` | responder-centering comparisons ("that happened to me too", "when i went through this") |
| `lecturing` | "what you need to realize", "the important thing is" |
| `performative_empathy` | "my heart goes out", "sending hugs", "i'm holding space for you" |
| `question_stacking` | 3+ question marks in one reply AND user emotion is negative, intensity ≥ 0.5 |

A failed check produces a `CheckResult` with a combined `critique` string. The pipeline wraps it in an `<empathy_retry>` system block (via `pipeline._critique_block`) that also quotes the first 200 chars of the failed draft, and re-calls `generate_reply` once. On success `trace.regenerated = True`; on failure or empty second draft, lemon ships the original.

---

## 9. Merged post-gen call — `empathy.post_exchange.bookkeep`

One LLM round-trip after the reply is delivered. Returns `(new_facts, nudged_state)`.

- Model: `STATE_MODEL`. `temperature=0.2`, `max_tokens=500`.
- Input: the user message + bot reply + existing facts dict + current state dict + last ~6 turns.
- Output: a single JSON object with `"facts"` and `"state"` sub-dicts.
- Parsing: strip fences → split → validate via `empathy.fact_extractor._validate` and `storage.state.validate_state`.
- Runs in a daemon thread from both `web.py::_stream_reply` and `lem.py::main`. See `session_context.run_bookkeeping`.
- Failure: returns `({}, current_state)` and logs to stdout. Never blocks or breaks the chat.

**The user never waits for this call.** The reply is delivered first; bookkeeping follows. On clean CLI exit, `lem.py` joins the final bookkeeping thread (timeout 10s) so the last state write isn't lost.

---

## 10. Web app — `web.py`

### 10.1 Startup

At import time the module:

1. Creates a `FastAPI` instance and a `threading.Lock()` for serialising access to the shared ChatContext.
2. Calls `db.start_session()` once, records the resulting `session_id` globally.
3. Loads the latest internal state via `storage.state.fresh_session_state()` (latest snapshot + session-start overrides).
4. Builds an initial history via `session_context.initial_history` (persona + time_context + internal_state + optional facts block).
5. Picks a random opener from `prompt.persona.LEMON_OPENERS`, appends it to `ctx.history` and logs it to `messages`.
6. Reads `templates/index.html` into `_INDEX_HTML` once (not on every `GET /`).

So the first-paint UI already has one assistant bubble in `/history` before the user sends anything.

### 10.2 Endpoints

| method | path | returns |
|---|---|---|
| GET  | `/`         | `templates/index.html` (cached at startup) |
| POST | `/chat`     | `text/event-stream` (SSE, see §10.3) |
| POST | `/command`  | `{"output": "...", "exit": bool}` |
| GET  | `/state`    | current internal state dict |
| GET  | `/facts`    | `{key: value, ...}` from `facts` table |
| GET  | `/sessions` | last 20 sessions with msg counts |
| GET  | `/history`  | non-system messages in the current in-memory session |
| GET  | `/trace`    | the last `PipelineTrace` serialised |
| GET  | `/docs`     | FastAPI's auto-generated OpenAPI explorer |

### 10.3 SSE protocol for `/chat`

Each event is `data: <json>\n\n`. The JSON always has `event` and `data`. Events:

| event | meaning |
|---|---|
| `phase` | pipeline phase label for the typing indicator: `"reading you"`, `"remembering"`, `"replying"`, `"rephrasing"` |
| `token` | the full reply body (delivered as one chunk after the pipeline completes) |
| `done`  | the full aggregated reply, emitted once at end |
| `error` | a string describing the upstream failure |

Phase events are buffered during `run_empathy_turn` and flushed before `token`. Consecutive duplicates are deduped so the UI doesn't see `"reading you"` twice when the pipeline fires it as its first phase.

Bookkeeping fires in a daemon thread **after** the `done` SSE event is yielded.

### 10.4 Frontend (`templates/index.html`)

Single HTML file, all CSS inline, vanilla JS. Key functions:

- `bubble(role, text)` renders one chat bubble; classes `you`, `lemon`, or `system`.
- `typingIndicator()` returns `{el, setPhase}`; `setPhase(p)` picks a random phrase from a flat `PHASE_PHRASES` list (~45 ice-cream-themed phrases) and shows `"lemon is <phrase>"` with a cross-fade animation.
- `streamChat(message)` POSTs to `/chat` and consumes the SSE stream. On the first `token` event, removes the typing indicator and opens a new lemon bubble. Splits on `\n\n` for iMessage-style multi-bubble replies.
- `loadState/loadFacts/loadSessions` populate the sidebar.
- `runCommand(text)` POSTs to `/command` and renders the response in a system bubble.

Light/dark theme toggle via CSS custom properties (`:root[data-theme="dark"]`) with the saved preference applied before body paint to avoid a flash.

### 10.5 Concurrency model

`threading.Lock()` wraps every read/write of the shared `ChatContext` on both sides:
- The SSE generator acquires the lock to snapshot `base_history`, then releases it during `run_empathy_turn`.
- The bookkeeping daemon thread acquires the same lock before mutating `ctx.internal_state` or calling `db.upsert_fact`.

Because the web server is meant for the user themselves (localhost), there is no per-user isolation. Running two browser tabs against the same server interleaves their messages into one conversation.

---

## 11. CLI — `lem.py`

REPL loop:

```
1. db.start_session(), fresh_session_state(), build initial history
2. print a random opener, log it
3. while not exit_requested:
     user_input = input("you: ")
     if is_command: dispatch; continue
     base_history = session_context.refresh_base_blocks(...)
     reply, trace = run_empathy_turn(user_input, base_history, ...)
     ctx.last_trace = trace
     append user + reply to ctx.history
     print(f"lemon: {reply}\n")
     spawn daemon thread: session_context.run_bookkeeping(...)
4. finally:
     last_bg.join(timeout=10)   # wait for last bookkeeping
     save_state + end_session(session_id)
```

Phase updates are printed inline as `  · reading you...`, `  · replying...` etc, via an `on_phase` callback.

---

## 12. Slash commands — `commands.py`

Every command is a function decorated with `@command("name", "help text")`. The decorator appends to the module-level `_REGISTRY`. `dispatch(text, ctx)` parses `/name arg...` and invokes the handler. Both the CLI and the `/command` endpoint call the same dispatcher; adding a new command works in both frontends with zero client-side changes.

Current commands:

| command | what it does |
|---|---|
| `/help` | list all commands |
| `/state` | print `ctx.internal_state` as JSON |
| `/reset` | reset internal state to defaults, save snapshot |
| `/facts` | list stored facts |
| `/remember key=value` | `db.upsert_fact(key, value, source_session_id=...)` |
| `/forget key` | `db.delete_fact(key)` |
| `/history [n]` | last `n*2` messages from `ctx.history` (non-system) |
| `/rewind` | pop the last two non-system messages from `ctx.history` |
| `/model name` | set `ctx.chat_model = name` for this session only |
| `/sessions` | `db.list_sessions(limit=10)` with msg counts |
| `/empathy [on\|off]` | mutate `config.ENABLE_EMPATHY_PIPELINE` |
| `/why` | render `ctx.last_trace` as a human-readable summary |
| `/quit` / `/exit` | set `ctx.exit_requested = True` (two decorators, one handler) |

`/why` renders the same data that `/trace` returns as JSON, but pretty-printed.

---

## 13. State layer

Three kinds of persistent state:

### 13.1 Internal state (`storage/state.py`)

Six-field dict. Rendered into `<internal_state>` every turn. Nudged every turn by the post-gen `bookkeep` call. Persisted as a row in `state_snapshots` whenever it changes.

| field | space |
|---|---|
| `mood` | neutral, good, low, happy, anxious, restless, tired, content |
| `energy` | low, medium, high |
| `engagement` | low, normal, deep |
| `emotional_thread` | free text or None, quiet background |
| `recent_activity` | free text or None, only when conversation grounds it |
| `disposition` | warm, normal, slightly reserved |

### 13.2 User facts

Key-value table. Upsert via `/remember` or `post_exchange.bookkeep`, delete via `/forget`. Rendered into the `<user_facts>` block every turn when non-empty. Facts persist across sessions.

### 13.3 Emotion-tagged messages

Every user message row gets `emotion`, `intensity`, and `salience` populated by the pipeline at log time (in `pipeline.run_empathy_turn`, right after the `user_read` call). Assistant rows leave those three columns NULL. `storage.memory.relevant_memories()` reads these rows via `db.find_messages_by_emotion`, filtered to user messages in other sessions.

---

## 14. Database layer — `storage/db.py`

See `db_schema.md` for column-by-column details.

- Four tables: `sessions`, `messages`, `state_snapshots`, `facts`. Plus `schema_version` for migration bookkeeping.
- Idempotent schema: `CREATE TABLE IF NOT EXISTS` on every connect.
- Migrations are a list of `(version, [SQL statements])`. On connect, any `version > current` is applied. "duplicate column name" errors are swallowed because fresh databases already have the column from the base SCHEMA.
- Every helper opens a short-lived connection via `@contextmanager connect()`.
- Every helper accepts an optional `path=` argument so tests can point at a per-test tmp file.

Access pattern (user-facing helpers):

```
start_session() → int
end_session(sid)
list_sessions(limit)

log_message(sid, role, content, emotion=, intensity=, salience=)
find_messages_by_emotion(emotion, exclude_session_id=, limit=)

latest_state() → dict | None
save_state_snapshot(state, session_id=)

upsert_fact(key, value, source_session_id=)
get_facts() → dict[str, str]
delete_fact(key)
```

---

## 15. History compression — `prompt/history.py`

Two helpers, both pure functions:

### `replace_system_block(history, tag, content, position)`

Drops any existing system message whose content contains `tag`, inserts a new one at `position`. Used by the base-block refresh each turn.

### `compress_history(history, keep_recent)`

Memory gradient. If conversation turns exceed `keep_recent`, older turns fold into a single system block tagged `<earlier_conversation>` with role-prefixed lines, inserted right after the leading system blocks. The `keep_recent` most recent turns stay verbatim.

---

## 16. Cost profile

Default knobs, Haiku everywhere:

| call | frequency | rough cost/turn |
|---|---|---|
| `user_read` (merged emotion + ToM) | every turn | ~$0.0002 |
| main chat (draft) | every turn | varies with context size; persona cached after turn 1 |
| main chat (retry) | only on post-check fail | same as draft |
| `bookkeep` (merged facts + state) | every turn, backgrounded | ~$0.0002 |

Per-turn ceiling with pipeline on: 3 Haiku calls (2 user-blocking). With `/empathy off`: 1 main call user-blocking + 1 backgrounded bookkeep.

---

## 17. Testing

Layout mirrors `src/`: one `tests/test_<module>.py` per source file. `tests/conftest.py` provides:

```python
@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
```

So every test gets a fresh SQLite file. LLM calls are not mocked at the HTTP level; the suite relies on pure-function tests for validators, formatters, history compression, detectors, time-context, and state CRUD. Coverage is partial; the new merged-call modules (`user_read`, `post_exchange`) don't yet have tests.

Run:

```bash
pip install -r requirements-dev.txt
pytest
pytest tests/test_empathy_check.py -v   # 46 tests, all green
```

---

## 18. Observability

Two introspection surfaces:

- **`/why`** (slash command): renders the last `PipelineTrace` as a human-readable summary.
- **`/trace`** (HTTP GET): returns the same trace as JSON.

`PipelineTrace` fields:

```python
emotion:        dict | None      # classifier output
memories:       list[dict]       # raw message rows retrieved
tom:            dict | None      # theory-of-mind output
draft:          str | None       # initial generation (before retry)
check:          CheckResult | None
regenerated:    bool
final:          str | None       # what lemon actually sent
pipeline_used:  bool             # False when ENABLE_EMPATHY_PIPELINE was off
facts_extracted: dict            # populated by the bookkeeping thread after the reply ships
```

The trace is attached to `ctx.last_trace` after every reply and persists only in memory. `facts_extracted` appears after a brief delay — bookkeeping runs post-reply.

---

## 19. Extending the system

### 19.1 Add a slash command

Drop this into `src/commands.py`:

```python
@command("mood", "force a mood: /mood happy")
def _mood(ctx: ChatContext, args: str) -> CommandResult:
    new = args.strip()
    if not new:
        return CommandResult(f"current mood: {ctx.internal_state['mood']}")
    ctx.internal_state["mood"] = new
    state_mod.save_state(ctx.internal_state, session_id=ctx.session_id)
    return CommandResult(f"mood forced to {new}.")
```

Available immediately in both CLI and web UI. No client-side changes.

### 19.2 Add a new system block

1. Write a formatter (pure function, takes data, returns `f"<mytag>...</mytag>"`). Place it in `prompt/` if it's refreshed per turn, or `empathy/` if it's pipeline-scoped.
2. Call it from `session_context.refresh_base_blocks`, or from the pipeline via `_inject_block`.

### 19.3 Add a pipeline step

Edit `pipeline.run_empathy_turn`. Steps are interchangeable: read → retrieve → inject → draft → check → optional retry. Adding a RAG step or best-of-N sampler means inserting a call and extending `PipelineTrace` with new fields.

### 19.4 Add a post-check detector

Append to `empathy.empathy_check.DETECTORS`:

```python
def _detect_my_thing(user_msg, draft, emotion):
    if ...:  return "critique text"
    return None

DETECTORS.append(("my_thing", _detect_my_thing))
```

### 19.5 Schema change

1. Add the column to the `SCHEMA` script.
2. Append `(N, ["ALTER TABLE ... ADD COLUMN ..."])` to `MIGRATIONS`.
3. Bump any helper that reads or writes the new column.

---

## 20. Known edges and limitations

- **Single-user, single-process.** `web.py` keeps one global `ChatContext`. Two tabs against the same server share one conversation.
- **No auth.** `web.py` is localhost-only by default (`host="127.0.0.1"`).
- **Bookkeeping lag.** Facts and state nudges appear in `/facts` and `/state` a few seconds after the reply (post-gen thread is async). `/trace.facts_extracted` is populated after that thread finishes.
- **SSE not resumable.** If the browser tab closes mid-stream the reply was still generated and logged; the user just doesn't see the replay.
- **Post-check is regex, not semantic.** Paraphrases of minimizing or validation-cascade phrases will still slip through. See `empathy_research.md` §2 for semantic alternatives.
- **Emotion classifier labels are coarse.** 21-label taxonomy trimmed from GoEmotions.
- **ToM pass doesn't see the memory block.** It consumes only the emotion read + last ~6 turns.
- **Some legacy test files are stale.** `test_emotion.py`, `test_fact_extractor.py`, `test_state.py`, `test_tom.py`, `test_chat.py`, plus parts of `test_pipeline.py` / `test_db.py` still reference pre-refactor symbols (`classify_emotion`, `update_internal_state`, etc.). They collect-error rather than run. The merged-call flow is covered by runtime smoke tests + `test_empathy_check.py` (46 tests).

---

## 21. Quick reference

**Start a chat from scratch:**

```bash
python src/lem.py          # CLI
python src/web.py          # web, open http://127.0.0.1:8000
```

**Inspect the database:**

```bash
sqlite3 .lemon.db
sqlite> SELECT role, content, emotion, intensity
   ...> FROM messages WHERE session_id = (SELECT MAX(id) FROM sessions)
   ...> ORDER BY id;
```

**See why lemon answered that way:**

```
/why
```

**Run with the pipeline off:**

```bash
LEMON_EMPATHY=0 python src/web.py
```

**Switch model for one session:**

```
/model anthropic/claude-sonnet-4.6
```

---

## 22. File-by-file cheat sheet

| file | purpose |
|---|---|
| `config.py` | env vars, model IDs, knobs, HTTP headers |
| `pipeline.py` | orchestrator: `read_user → memory → draft → check → regen-once` |
| `session_context.py` | `initial_history`, `refresh_base_blocks`, `run_bookkeeping` — shared CLI+web |
| `commands.py` | slash-command registry + 12 built-ins |
| `lem.py` | CLI REPL |
| `web.py` | FastAPI app + SSE + introspection |
| `prompt/persona.py` | persona prompt + opener pool |
| `prompt/time_context.py` | `<time_context>` block generator |
| `prompt/history.py` | `replace_system_block` + `compress_history` |
| `prompt/facts.py` | `<user_facts>` block formatter |
| `empathy/emotion.py` | emotion schema, `_validate`, `format_emotion_block` |
| `empathy/tom.py` | ToM schema, `_validate`, `format_tom_block` |
| `empathy/fact_extractor.py` | fact-key regex + `_validate` |
| `empathy/empathy_check.py` | 12-detector post-check |
| `empathy/user_read.py` | merged pre-gen LLM call |
| `empathy/post_exchange.py` | merged post-gen LLM call |
| `llm/chat.py` | OpenRouter reply call, cache wrap, streaming |
| `llm/parse_utils.py` | shared fence-stripper + recent-msgs formatter |
| `storage/db.py` | schema, migrations, CRUD helpers |
| `storage/memory.py` | emotion-tagged retrieval + `<emotional_memory>` formatter |
| `storage/state.py` | internal state: defaults, load/save/format + validator |
| `templates/index.html` | single-page web UI |
