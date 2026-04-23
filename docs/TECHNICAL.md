# Technical Reference

A single-document walkthrough of lemon's implementation: every module, every call path, every piece of state, every request shape.

This complements rather than replaces the existing docs. For narrative context see `architecture.md`. For clinical empathy theory see `empathy_research.md`. For schema details see `db_schema.md`. This file is the "how it actually runs" reference.

---

## 1. Overview

Lemon is a chat assistant styled as a friend, not a productivity tool. Implementation is a small Python codebase (~1.5k LoC in `src/`) with two frontends over one backend:

- **Backend core:** per-turn "empathy pipeline" that runs four cheap LLM calls plus a regex-based post-check around a single main generation call.
- **CLI:** `src/lem.py`, a stdin/stdout REPL with humanized output pacing.
- **Web:** `src/web.py`, FastAPI + a single hand-written HTML page + Server-Sent Events for streaming.
- **Persistence:** one SQLite file with four tables (sessions, messages, state_snapshots, facts). Idempotent schema + migration table.
- **Model layer:** OpenRouter as the HTTP target, Anthropic Claude Haiku 4.5 as the default for both main chat and auxiliary calls. Anthropic-style `cache_control` breakpoints on the persona block when the model supports it.

Everything is synchronous. One process, one conversation at a time. No auth, no multi-user.

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
| `LEMON_STATE_MODEL` | `anthropic/claude-haiku-4.5` | emotion classifier, ToM, state updater |
| `LEMON_PROMPT_CACHE` | auto (`1` for `anthropic/*`, else `0`) | wrap persona block in `cache_control: ephemeral` |
| `LEMON_EMPATHY` | `1` | master switch for the pipeline |
| `LEMON_EMPATHY_RETRY` | `1` | regenerate once when the post-check fails |
| `LEMON_MEMORY_LIMIT` | `3` | how many matching-emotion memories to inject |
| `LEMON_HUMANIZE` | `1` | per-token typing pacing |
| `LEMON_DB` | `.lemon.db` | SQLite file path (gitignored) |

### 2.3 In-code constants (not env-settable)

From `src/config.py`:

```python
STATE_UPDATE_EVERY  = 2         # run state updater every N exchanges
KEEP_RECENT_TURNS   = 8         # before compress_history folds older turns
HUMANIZE_BASE_SECONDS = 0.018   # base per-token delay, scaled by energy
HUMANIZE_PUNCT_PAUSE  = 0.18    # extra pause after . ! ? ,
```

From `src/chat.py`:

```python
ENERGY_SPEED_MULT = {"low": 1.45, "medium": 1.0, "high": 0.7}   # state → pacing
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

Every LLM call in the codebase uses this single endpoint and header dict.

---

## 3. Module map

```
src/
  config.py         env, models, paths, knobs, HTTP headers
  prompt.py         persona system-prompt string + opener pool
  time_context.py   <time_context> block generator
  facts.py          <user_facts> block formatter
  state.py          internal state: load, save, format, parse, update
  history.py        replace_system_block + compress_history
  db.py             SQLite layer: schema, migrations, all read/write helpers
  chat.py           OpenRouter call, cache wrap, streaming, humanize pacing
  emotion.py        pre-generation user-emotion classifier
  tom.py            theory-of-mind side pass
  memory.py         emotion-tagged memory retrieval + formatting
  empathy_check.py  regex-based post-check (5 detectors)
  pipeline.py       orchestrator: the "empathy pipeline"
  commands.py       slash-command registry + ChatContext + dispatcher
  lem.py            CLI REPL entry point
  web.py            FastAPI app + SSE + introspection endpoints
  templates/
    index.html      single-page web UI (vanilla JS, inline CSS)
tests/              one test file per src/ module + conftest.py
docs/               architecture, slash commands, db schema, web ui, empathy research
```

Dependency direction flows left-to-right:

```
config ──► everything
db ──► state, memory, commands, pipeline, web.py, lem.py
chat ──► pipeline, web.py, lem.py
emotion + tom + memory + empathy_check ──► pipeline
history + prompt + time_context + facts + state ──► web.py, lem.py
commands ──► lem.py, web.py
pipeline ──► lem.py, web.py
```

There are no cycles. Tests are fully isolated via an autouse `isolated_db()` fixture in `tests/conftest.py`.

---

## 4. Per-turn request lifecycle

This is the full path a single user message takes. Numbered callouts are the LLM calls.

```
user types a message
  │
  ▼
[refresh base blocks]  CLI: lem.refresh_base_blocks()
                       web: _refresh_base_blocks()
  replaces <time_context>, <internal_state>, <user_facts>
  system blocks (positions 1, 2, 3) via history.replace_system_block.
  │
  ▼
if is_command(user_input):
    dispatch(user_input, ctx) ──► system bubble; no LLM call; return
  │
  ▼
pipeline.run_empathy_turn(user_msg, base_history, ...)
  │
  ├── if ENABLE_EMPATHY_PIPELINE is False:
  │     append user_msg + compress_history + (1) generate_reply + return
  │
  ├── recent = _recent_messages_for_context(base_history, n=6)
  │
  ├── (1) emotion = emotion.classify_emotion(user_msg, recent)       [STATE_MODEL]
  │       returns {primary, intensity, undertones, underlying_need}
  │       phase: "reading you"
  │       db.log_message(user_msg, emotion=..., intensity=..., salience=intensity)
  │
  ├── memories = memory.relevant_memories(emotion.primary,           [SQLite only]
  │                  current_session_id, limit=MEMORY_RETRIEVAL_LIMIT)
  │       phase: "remembering"
  │       skipped when primary == "neutral"
  │
  ├── (2) tom = tom.theory_of_mind(user_msg, emotion, recent)         [STATE_MODEL]
  │       returns {feeling, avoid, what_helps}
  │       phase: "thinking"
  │
  ├── history = base_history, then:
  │     - _inject_block(<emotional_memory>, memories)   if memories
  │     - _inject_block(<user_emotion>, emotion_block)
  │     - _inject_block(<theory_of_mind>, tom_block)
  │     each insertion drops any prior block with the same tag and
  │     re-inserts it right after the leading contiguous system run.
  │
  ├── history.append({role: user, content: user_msg})
  ├── history = history.compress_history(history, keep_recent=8)
  │
  ├── (3) draft = chat.generate_reply(history, model)                 [CHAT_MODEL]
  │       phase: "replying"
  │       buffered (non-streaming from caller's POV) because the
  │       post-check needs the full reply before anything is shown.
  │
  ├── check = empathy_check.check_response(user_msg, draft, emotion)
  │       five regex detectors. Pure Python, no LLM call.
  │
  ├── final = draft
  │   if not check.passed and EMPATHY_RETRY_ON_FAIL:
  │       phase: "rephrasing"
  │       retry_history = _inject_block(<empathy_retry>, critique_block)
  │       (4) second = chat.generate_reply(retry_history)            [CHAT_MODEL]
  │       if second.strip(): final = second; trace.regenerated = True
  │
  └── db.log_message(assistant, final); return (final, trace)
  │
  ▼
caller (web._stream_reply or lem.main):
  - ctx.last_trace = trace
  - append user_msg + final to ctx.history
  - replay `final` token-by-token via re_split_keep_whitespace, sleeping
    humanize_delay(token, energy) between chunks.
    CLI: print to stdout; web: yield SSE "token" event.
  - every STATE_UPDATE_EVERY exchanges:
      (5) ctx.internal_state = state.update_internal_state(state, user, reply)  [STATE_MODEL]
      db.save_state_snapshot(...)
```

**Total per turn:** 3 LLM calls guaranteed (emotion, ToM, main), +1 if `/empathy off` was toggled (just main), +1 if the post-check fails (retry), +1 every second turn (state updater). All auxiliary calls use `STATE_MODEL`; only the main draft and retry use `CHAT_MODEL`.

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

- **Order of injection is deterministic.** `_inject_block` drops any prior block with the same tag, then inserts the new content just after the leading contiguous block of system messages. So re-running the pipeline on the same base history produces the same shape.
- **`time_context` + `internal_state` + `user_facts` change every turn** and by construction live at positions 1, 2, 3 (managed by `replace_system_block` in `history.py`).
- **Persona block never changes.** That is what makes the cache hit work.
- **Emotion/ToM/Memory blocks are pipeline-scoped.** They live inside the history temporarily for one call, then are dropped when the pipeline rebuilds next turn.

### 5.1 Persona block (`prompt.LEMON_PROMPT`)

Sections inside `<Who you are>` through `<forbidden words>`. The model is told explicitly never to emit the tag names. Notable sections:

- `<Who you are>`: role definition (friend, not assistant, no gender).
- `<Voice and tone rules>`: WhatsApp-register constraints.
- `<internal_state_instructions>`: tells the model how to read the state block (not to narrate it, let it leak through).
- `<time_aware_personality>`: tone rules per time-of-day and session length.
- `<time_awareness>`: anticipation-nudge rule for upcoming events the user mentioned.
- `<rules_of_time>`: irreversibility, causality, duration-asymmetry axioms.
- `<language mirroring>`: Hinglish-default, match user's language.
- `<conversation rules>`: "only respond to what the user said", no unsolicited advice, short by default.
- `<formatting>`: no hyphens, no lists, keep messages flowing.
- `<forbidden words>`: negative-list ("Vibe", "quiet", "great to see you", "What is on your mind").

The block is ~5KB. It never changes between turns, so Anthropic-style prompt caching on the persona block is effectively free after the first call.

### 5.2 Refreshed-per-turn blocks

#### `<time_context>` (`time_context.get_time_context`)

```
<time_context>
Current local date: 2026-04-23
Current local time: 15:42
Day of week: Thursday
Time of day: afternoon
You've been talking for a bit now, around 18 minutes.
</time_context>
```

Time buckets: morning (5-9), afternoon (10-16), evening (17-20), late night (21-23), very late night (0-4). Duration bucket text depends on elapsed minutes.

#### `<internal_state>` (`state.format_internal_state`)

Rendered from the 6-field dict:

```
<internal_state>
...
Mood: content
Energy: medium
Engagement level: deep
What's on your mind: curious about exam result
What you've been up to: nothing worth mentioning
Disposition toward this person right now: warm
...
</internal_state>
```

Defaults in `state.DEFAULT_STATE`:

```python
{"mood": "neutral", "energy": "medium", "engagement": "normal",
 "emotional_thread": None, "recent_activity": None, "disposition": "warm"}
```

#### `<user_facts>` (`facts.format_user_facts`)

```
<user_facts>
...
  college_year: second
  city: Bangalore
...
</user_facts>
```

Skipped entirely when the `facts` table is empty.

### 5.3 Pipeline-injected blocks

#### `<user_emotion>` (`emotion.format_emotion_block`)

```
<user_emotion>
Primary feeling: sadness (moderate, intensity 0.52)
Undertones: loneliness, tired
What they probably want: feel heard, not solved
Let this shape your tone, length, and whether to ask vs. acknowledge. Do not echo the label back.
</user_emotion>
```

Intensity word ladder: `<0.3 mild`, `<0.6 moderate`, `<0.85 strong`, else `very strong`.

#### `<theory_of_mind>` (`tom.format_tom_block`)

```
<theory_of_mind>
What they're feeling: tired and a little embarrassed about the exam
Don't: don't jump to advice
Do: stay with it, ask one open question if anything
</theory_of_mind>
```

#### `<emotional_memory>` (`memory.format_memory_block`)

```
<emotional_memory>
- yesterday, when feeling sadness: "i just feel flat about this..."
- 4 days ago, when feeling sadness: "everyone's moved on and I'm still stuck"
</emotional_memory>
```

Timestamps are humanized via `memory._humanize_age`: `today`, `yesterday`, `N days ago`, `N weeks ago`, `N months ago`.

---

## 6. The three auxiliary LLM calls

All three go through OpenRouter with `STATE_MODEL` (Haiku by default), `temperature` and `max_tokens` tuned per task. All three tolerate failure by returning a safe default rather than propagating an exception, so a dropped auxiliary call downgrades the run silently rather than breaking the chat.

### 6.1 Emotion classifier (`emotion.classify_emotion`)

- Model call params: `temperature=0.2`, `max_tokens=250`.
- Labels: 21-label set (neutral, joy, excitement, love, gratitude, sadness, loneliness, disappointment, grief, anger, frustration, annoyance, fear, anxiety, confusion, shame, embarrassment, guilt, tired, amused, curious).
- Parser tolerates fenced code blocks.
- Fallback: `{"primary":"neutral","intensity":0.3,"underlying_need":None,"undertones":[]}`.
- Side effect: the user message row in `messages` is written with `emotion`, `intensity`, and `salience=intensity` fields populated.

### 6.2 Theory of mind (`tom.theory_of_mind`)

- Model call params: `temperature=0.4`, `max_tokens=350`.
- Consumes the emotion output + last ~6 non-system turns.
- Output shape: `{feeling, avoid, what_helps}`, each a short string or `None`.
- Fallback: all three fields `None`.
- Phrased internally as "quiet observer" so the model does not draft a reply.

### 6.3 State updater (`state.update_internal_state`)

- Model call params: `temperature=0.3`, `max_tokens=400`.
- Runs only every `STATE_UPDATE_EVERY` exchanges (default 2).
- Prompt emphasises "subtle nudges, not dramatic shifts". Mood and energy shift slowly, `recent_activity` is only set when the conversation causally grounds it.
- Failure: return the old state unchanged. Chat never dies because the state updater choked.
- Persisted via `db.save_state_snapshot` (new row per update).

---

## 7. Main chat call (`chat.py`)

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

Temperature + top_p come from the v2 tuning pass (see `temporal_reasoning.txt`): 0.75 produces natural variance without going off-prompt. `frequency_penalty=0.2` prevents "I hear you, that's so valid" style phrase-stacking.

### 7.2 Prompt caching (`prepare_messages`)

Only runs if `ENABLE_PROMPT_CACHE` is true. It wraps the persona block into Anthropic's structured-content form:

```python
{"role": "system",
 "content": [{"type": "text", "text": content,
              "cache_control": {"type": "ephemeral"}}]}
```

Any system message containing `<Who you are>` (matched by `PERSONA_TAG`) is wrapped. Everything else stays as a plain string, so other system blocks go uncached. After the first call, the persona is a cache hit on the Anthropic side, so subsequent turns pay roughly zero cost for those 5KB of prompt.

### 7.3 Streaming vs buffering

Two paths in `chat.py`:

- **`generate_reply`** (buffered): calls `iter_chat` and concatenates all deltas. Used by the empathy pipeline's draft + retry phases so the post-check can run on a complete string.
- **`stream_chat`** (streamed): for CLI paths that bypass the pipeline. Prints each delta as it arrives.

The web UI combines both: the backend buffers to run the check, then re-streams the final reply back to the browser token-by-token via SSE with pacing (see §9.3).

### 7.4 Humanize pacing (`humanize_delay`)

Per-token delay in seconds:

```
base  = HUMANIZE_BASE_SECONDS * ENERGY_SPEED_MULT[energy]      # 0.018 × {1.45, 1.0, 0.7}
delay = base * uniform(0.5, 1.5)
if token ends in ".!?," : delay += HUMANIZE_PUNCT_PAUSE * mult * uniform(0.7, 1.3)
```

So a tired (`energy: low`) lemon types ~45% slower, and punctuation adds a ~180 ms pause scaled by the same factor. Disabled entirely when `LEMON_HUMANIZE=0`.

### 7.5 Tokenisation for replay

`re_split_keep_whitespace(text)` splits on whitespace and keeps trailing whitespace attached to the prior chunk:

- Input: `"hey that's rough.\n\ndo you want to talk?"`
- Output: `["hey ", "that's ", "rough.\n\n", "do ", "you ", "want ", "to ", "talk?"]`

Two places consume this: `chat.play_tokens` (CLI) and `web._stream_reply` (SSE). Because the frontend splits bubbles on `\n\n`, it matters that whitespace stays attached to the prior chunk so `\n\n` arrives as one SSE `token` event, not split across two.

---

## 8. Empathy post-check (`empathy_check.py`)

Pure Python, regex-based, zero LLM cost. Runs after the main draft. Five detectors, each returns a critique string or `None`:

| detector | fires when | example trigger |
|---|---|---|
| `minimizing` | draft matches `MINIMIZING` regex | "at least", "could be worse", "you'll be fine", "just a..." |
| `toxic_positivity` | draft matches `TOXIC_POSITIVITY` regex | "positive vibes", "stay strong", "silver lining", "this too shall pass" |
| `advice_pivot` | draft starts with `ADVICE_PIVOT` regex **and** user emotion in `NEGATIVE_EMOTIONS` **and** intensity ≥ 0.5 | "you should", "try to", "why don't you", "have you tried" |
| `polarity_mismatch` | draft starts with `CHEERY_OPENERS` **and** user emotion in `NEGATIVE_EMOTIONS` **and** intensity ≥ 0.4 | opens "haha", "nice", "sweet", "awesome" while user is sad |
| `validation_cascade` | draft contains ≥ 3 `CASCADE_PHRASES` | "I hear you... that makes sense... your feelings are valid..." |

`NEGATIVE_EMOTIONS = {sadness, loneliness, disappointment, grief, anger, frustration, annoyance, fear, anxiety, shame, embarrassment, guilt, tired}`.

A failed check produces a `CheckResult` with a combined `critique` string. The pipeline wraps it in an `<empathy_retry>` system block (via `pipeline._critique_block`) that also quotes the first 200 chars of the failed draft, and re-calls `generate_reply` once. On success `trace.regenerated = True`; on failure or empty second draft, lemon ships the original.

---

## 9. Web app (`web.py`)

### 9.1 Startup

At import time the module:

1. Creates a `FastAPI` instance and a `threading.Lock()` for serialising access to the shared ChatContext.
2. Calls `db.start_session()` once, records the resulting `session_id` globally.
3. Loads the latest internal state via `state.load_state()`.
4. Builds an initial history (persona + time_context + internal_state + optional facts block).
5. Picks a random opener from `prompt.LEMON_OPENERS`, appends it to `ctx.history` and logs it to `messages`.

So the first-paint UI already has one assistant bubble in `/history` before the user sends anything.

### 9.2 Endpoints

| method | path | returns |
|---|---|---|
| GET  | `/`         | `templates/index.html` |
| POST | `/chat`     | `text/event-stream` (SSE, see §9.3) |
| POST | `/command`  | `{"output": "...", "exit": bool}` |
| GET  | `/state`    | current internal state dict |
| GET  | `/facts`    | `{key: value, ...}` from `facts` table |
| GET  | `/sessions` | last 20 sessions with msg counts |
| GET  | `/history`  | non-system messages in the current in-memory session |
| GET  | `/trace`    | the last `PipelineTrace` serialised (emotion, tom, memories count, check result, regenerated flag) |
| GET  | `/docs`     | FastAPI's auto-generated OpenAPI explorer |

`/chat` and `/command` are strict about what they accept: `/chat` rejects slash commands with HTTP 400, `/command` rejects non-slash input the same way.

### 9.3 SSE protocol for `/chat`

Each event is a single line: `data: <json>\n\n`. The JSON always has two fields, `event` and `data`. Events:

| event | meaning |
|---|---|
| `phase` | pipeline phase label for the typing indicator: `"reading you"`, `"remembering"`, `"thinking"`, `"replying"`, `"rephrasing"` |
| `token` | a content delta to append to the current assistant bubble |
| `done`  | the full aggregated reply, emitted once at end |
| `error` | a string describing the upstream failure |

Timing shape: phases are yielded first in bursts (the pipeline finishes server-side before any token goes out); tokens then stream at humanized pace via `time.sleep(humanize_delay(chunk, energy))` between chunks.

### 9.4 Frontend (`templates/index.html`)

Single HTML file, all CSS inline, vanilla JS (no bundler). Key functions:

- `bubble(role, text)` renders one chat bubble; classes `you`, `lemon`, or `system`.
- `typingIndicator()` returns `{el, setPhase}`; `setPhase(s)` rewrites the label to `"lemon is " + s`.
- `streamChat(message)` POSTs to `/chat` and consumes the SSE stream. On the first `token` event, removes the typing indicator and opens a new lemon bubble. On every `token` event, splits on `\n\n` and opens a new bubble at each paragraph break (iMessage-style multi-bubble replies). Scrolls to bottom on each token.
- `loadHistory()` populates on page load, splitting any past multi-paragraph assistant message on `\n\n` for visual consistency.
- `loadState/loadFacts/loadSessions` populate the sidebar.
- `runCommand(text)` POSTs to `/command` and renders the response in a system bubble, then refreshes the sidebar.
- `send()` routes `/`-prefixed input to `runCommand`, anything else to `streamChat`.

Styling uses CSS custom properties (`--bg`, `--chat-bg`, `--bubble-you`, `--bubble-lemon`, etc.), with a `prefers-color-scheme: dark` media query swapping the palette.

### 9.5 Concurrency model

`threading.Lock()` wraps every read/write of the shared `ChatContext`. A single `_exchange_count` module-global increments after each reply so the state updater runs on the correct cadence. The pipeline itself does not hold the lock: it receives an immutable snapshot of `base_history` and returns `(reply, trace)`. The lock re-engages only to append to `ctx.history` and run the state updater.

Because the web server is meant for the user themselves (localhost), there is no per-user isolation. Running two browser tabs against the same server interleaves their messages into one conversation.

---

## 10. CLI (`lem.py`)

REPL loop:

```
1. db.start_session(), load_state(), build initial history
2. print a random opener, log it
3. while not exit_requested:
     user_input = input("you: ")
     if is_command: dispatch; continue
     base_history = refresh_base_blocks(ctx, session_start)
     reply, trace = run_empathy_turn(user_input, base_history, ...)
     ctx.last_trace = trace
     append user + reply to ctx.history
     play_tokens(reply, energy=state.energy)     # stdout with humanize pacing
     every STATE_UPDATE_EVERY: update_internal_state + save_state
4. finally: save_state + end_session(session_id)
```

Phase updates are printed inline as `  · reading you...`, `  · thinking...` etc, via an `on_phase` callback.

---

## 11. Slash commands (`commands.py`)

Every command is a function decorated with `@command("name", "help text")`. The decorator appends to the module-level `_REGISTRY`. `dispatch(text, ctx)` parses `/name arg...` and invokes the handler. Both the CLI and `/command` endpoint call the same dispatcher; adding a new command works in both frontends with zero client-side changes.

Current commands:

| command | what it does |
|---|---|
| `/help` | list all commands |
| `/state` | print `ctx.internal_state` as JSON |
| `/reset` | reset internal state to defaults, save snapshot; does not erase facts or history |
| `/facts` | list stored facts |
| `/remember key=value` | `db.upsert_fact(key, value, source_session_id=...)` |
| `/forget key` | `db.delete_fact(key)` |
| `/history [n]` | last `n*2` messages from `ctx.history` (non-system), default n=5 |
| `/rewind` | pop the last two non-system messages from `ctx.history` |
| `/model name` | set `ctx.chat_model = name` for this session only |
| `/sessions` | `db.list_sessions(limit=10)` with msg counts |
| `/empathy [on\|off]` | mutate `config.ENABLE_EMPATHY_PIPELINE`; no arg prints current status |
| `/why` | render `ctx.last_trace` as a human-readable summary |
| `/quit` / `/exit` | set `ctx.exit_requested = True` |

`/why` renders the same data that `/trace` returns as JSON, but pretty-printed. Example output:

```
last reply's pipeline trace:
  emotion: sadness (intensity 0.65)
           undertones: loneliness, tired
           underlying need: feel heard, not solved
  feeling:   tired and let down about the exam
  avoid:     don't jump to advice
  do:        stay with it, ask one open question
  memories used: 2
  post-check: passed
```

---

## 12. State layer

Three kinds of persistent state:

### 12.1 Internal state (`state.py`)

Six-field dict. Rendered into `<internal_state>` every turn. Nudged every `STATE_UPDATE_EVERY` exchanges by the state-updater LLM call. Persisted as a row in `state_snapshots` every time it changes. `state.load_state()` merges the saved snapshot with defaults so new fields added in code don't break old databases.

| field | type | space |
|---|---|---|
| `mood` | string | neutral, good, low, happy, anxious, restless, tired, content |
| `energy` | string | low, medium, high |
| `engagement` | string | low, normal, deep |
| `emotional_thread` | string or None | free text, quiet background |
| `recent_activity` | string or None | free text, only when conversation grounds it |
| `disposition` | string | warm, normal, slightly reserved |

### 12.2 User facts

Key-value table. Upsert via `/remember`, delete via `/forget`. Rendered into the `<user_facts>` block every turn when non-empty. Facts persist across sessions.

### 12.3 Emotion-tagged messages

Every user message row gets `emotion`, `intensity`, and `salience` populated by the pipeline at log time (inside `pipeline.run_empathy_turn`, right after the classifier call). Assistant rows leave those three columns NULL. The emotion-tagged rows are what `memory.relevant_memories()` reads from via `db.find_messages_by_emotion`, filtered to user messages in other sessions.

---

## 13. Database layer (`db.py`)

See `db_schema.md` for column-by-column details. Summary:

- Four tables: `sessions`, `messages`, `state_snapshots`, `facts`. Plus `schema_version` for migration bookkeeping.
- Idempotent schema: `CREATE TABLE IF NOT EXISTS` on every connect.
- Migrations are a list of `(version, [SQL statements])`. On connect, any `version > current` is applied. "duplicate column name" errors are swallowed because fresh databases already have the column from the base SCHEMA.
- Every helper opens a short-lived connection via `@contextmanager connect()`, runs one statement set, commits, closes. Foreign keys on, WAL off.
- Every helper accepts an optional `path=` argument so tests can point at a per-test tmp file. The `isolated_db()` autouse fixture in `conftest.py` monkeypatches `config.DB_PATH` to a `tmp_path` per test.

Access pattern (user-facing helpers, all in `db.py`):

```
start_session() → int
end_session(sid)
list_sessions(limit)

log_message(sid, role, content, emotion=, intensity=, salience=)
session_messages(sid)
find_messages_by_emotion(emotion, exclude_session_id=, limit=)
find_recent_messages(limit=)

latest_state() → dict | None
save_state_snapshot(state, session_id=)

upsert_fact(key, value, source_session_id=)
get_facts() → dict[str, str]
delete_fact(key)
clear_facts()
```

---

## 14. History compression (`history.py`)

Two helpers, both pure functions:

### `replace_system_block(history, tag, content, position)`

Drops any existing system message whose content contains `tag`, inserts a new one at `position`. Used by the CLI and web-UI base-block refresh to rewrite the `<time_context>`, `<internal_state>`, and `<user_facts>` blocks each turn without letting them accumulate.

### `compress_history(history, keep_recent)`

Memory gradient. Separates system vs conversation messages. If conversation turns exceed `keep_recent`, everything older is joined into a single system block tagged `<earlier_conversation>` with role-prefixed lines, inserted right after the leading system blocks. The `keep_recent` most recent turns stay verbatim. Below threshold, the history passes through unchanged.

Format of the summary block:

```
<earlier_conversation>
Here is a rough record of what was said earlier in this chat. It is not recent
but it is part of your shared history with this person. Reference it only if
it comes up naturally, not to fill silence.

User: ...
Assistant: ...
User: ...
</earlier_conversation>
```

Runs once per turn, right before the main generation call.

---

## 15. Cost profile

Default knobs, Haiku everywhere:

| call | frequency | rough cost/turn |
|---|---|---|
| emotion classifier | every turn | ~$0.0001 |
| ToM pass | every turn | ~$0.0002 |
| main chat (draft) | every turn | varies with context size; persona cached after turn 1 |
| main chat (retry) | only on post-check fail | same as draft |
| state updater | every `STATE_UPDATE_EVERY` turns (default 2) | ~$0.0001 |

Per-turn ceiling with pipeline on: ~3 auxiliary Haiku calls + 1 main Haiku call. With `/empathy off`: 1 main call. No fine-tuning, no local inference, no GPU at all.

---

## 16. Testing

Layout mirrors `src/`: one `tests/test_<module>.py` per source file. `tests/conftest.py` provides:

```python
@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point config.DB_PATH at a per-test temporary file."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
```

So every test gets a fresh SQLite file with the schema already migrated. LLM calls are not mocked at the HTTP level; they are avoided by testing pure functions (`compress_history`, `format_emotion_block`, `parse_state_response`, `check_response`, `humanize_delay`, `time_of_day_label`, etc.) and by injecting fake response dicts where the shape matters.

Run:

```bash
pip install -r requirements-dev.txt
pytest               # all
pytest tests/test_pipeline.py -v
```

---

## 17. Observability

Two introspection surfaces:

- **`/why`** (slash command): renders the last `PipelineTrace` as a human-readable summary. Available in both CLI and web UI.
- **`/trace`** (HTTP GET): returns the same trace as JSON. Useful for the web UI to show a debug panel, or for external tooling.

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
```

The trace is attached to `ctx.last_trace` after every reply and persists only in memory (not written to the DB).

---

## 18. Extending the system

### 18.1 Add a slash command

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

### 18.2 Add a new system block

1. Write a formatter (pure function, takes data, returns `f"<mytag>...</mytag>"`).
2. Call it from `refresh_base_blocks` in both `lem.py` and `web.py`, or from the pipeline if it's empathy-pass-scoped, and feed it through `replace_system_block` or `_inject_block`.
3. Pick a position: constant tags (refreshed each turn) live at 1-3; pipeline tags live after those.

### 18.3 Add a pipeline step

Edit `pipeline.run_empathy_turn`. Steps are short and interchangeable: classify → retrieve → ToM → inject → draft → check → optional retry. Adding a best-of-N sampler or a RAG step means inserting a call in the right place and extending `PipelineTrace` with new fields. `/why` and `/trace` pick up new trace fields automatically as long as you use `getattr(trace, ..., default)`.

### 18.4 Add a post-check detector

Append to `empathy_check.DETECTORS`:

```python
def _detect_my_thing(user_msg, draft, emotion):
    if ...:  return "critique text"
    return None

DETECTORS.append(("my_thing", _detect_my_thing))
```

No other changes needed. The critique is aggregated into the regenerate prompt automatically.

### 18.5 Schema change

1. Add the column to the `SCHEMA` script (so fresh DBs get it directly).
2. Append `(N, ["ALTER TABLE ... ADD COLUMN ..."])` to `MIGRATIONS`.
3. `LATEST_VERSION` is computed via `max(...)`, so it updates automatically.
4. Bump any helper that reads or writes the new column.

---

## 19. Known edges and limitations

- **Single-user, single-process.** `web.py` keeps one global `ChatContext`. Two tabs against the same server share one conversation.
- **No auth.** `web.py` is localhost-only by default (`host="127.0.0.1"`). Exposing it needs a reverse proxy + auth layer.
- **SSE not resumable.** If the browser tab closes mid-stream the full reply was still generated server-side and logged to `messages`, so next `/history` call will include it. The user just does not see the tokens replay.
- **Pipeline latency.** Three extra calls add ~1-2s to perceived latency. Phase labels soften this but it is real.
- **Post-check is regex, not semantic.** Paraphrases of minimizing or validation-cascade phrases will slip through. Mitigations documented in `empathy_research.md` §2 (classifier-guided decoding, best-of-N judge) would tighten this.
- **Emotion classifier labels are coarse.** 21-label taxonomy trimmed from GoEmotions. Fine-grained subtypes (`relief`, `contempt`, `pride`, etc.) are not distinguished.
- **ToM pass does not see the memory block.** It consumes only the emotion + last ~6 turns. Feeding retrieved memories into ToM might sharpen it, at the cost of one more injection step.
- **State updater runs on every second turn unconditionally.** Even when the exchange was trivial small-talk, we still pay the call. A cheap "is this exchange state-relevant?" filter would cut cost.

---

## 20. Quick reference

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

**Per-test DB isolation:**

```python
def test_my_thing(isolated_db):   # autouse fixture kicks in
    sid = db.start_session()
    ...
```

---

## 21. File-by-file cheat sheet

| file | ~LoC | purpose |
|---|---|---|
| `config.py` | 50 | env vars, model IDs, knobs, HTTP headers |
| `prompt.py` | 125 | persona prompt + opener pool |
| `time_context.py` | 40 | `<time_context>` block generator |
| `facts.py` | 20 | `<user_facts>` block formatter |
| `state.py` | 120 | internal state CRUD + LLM updater |
| `history.py` | 40 | `replace_system_block` + `compress_history` |
| `db.py` | 280 | schema, migrations, CRUD helpers |
| `chat.py` | 160 | OpenRouter call, cache wrap, humanize pacing |
| `emotion.py` | 170 | classifier + `<user_emotion>` formatter |
| `tom.py` | 140 | ToM pass + `<theory_of_mind>` formatter |
| `memory.py` | 70 | emotion-tagged retrieval + `<emotional_memory>` formatter |
| `empathy_check.py` | 170 | five post-check detectors |
| `pipeline.py` | 200 | orchestrator: classify → retrieve → ToM → draft → check → retry |
| `commands.py` | 220 | slash-command registry + 13 built-ins |
| `lem.py` | 120 | CLI REPL |
| `web.py` | 255 | FastAPI app + SSE + introspection |
| `templates/index.html` | ~270 | single-page web UI |

Total: ~2.5k LoC including tests.
