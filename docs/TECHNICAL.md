# Technical Reference

A single-document walkthrough of lemon's implementation: every module, every call path, every piece of state, every request shape.

This complements rather than replaces the existing docs. For narrative context see `architecture.md`. For clinical empathy theory see `empathy_research.md`. For schema details see `db_schema.md`. This file is the "how it actually runs" reference.

---

## 1. Overview

Lemon is a chat assistant styled as a friend, not a productivity tool. Implementation is a small Python codebase in `src/` with two frontends over one backend:

- **Backend core:** per-turn "empathy pipeline" that runs one merged pre-gen LLM call (4 sub-objects: emotion + ToM + user_state_delta + lemon_state_delta), one main reply call (+ optional retry), and one merged post-gen LLM call (facts-only since stage 2 of the dyadic-state work).
- **CLI:** `src/lem.py`, a stdin/stdout REPL.
- **Web:** `src/web.py`, FastAPI + a single hand-written HTML page + Server-Sent Events for streaming.
- **Persistence:** one SQLite file. Six tables: `sessions`, `messages`, `state_snapshots` (legacy archive), `lemon_state_snapshots`, `user_state_snapshots`, `facts`. Idempotent schema + migration table at version 3.
- **Model layer:** OpenRouter as the HTTP target, Anthropic Claude Haiku 4.5 as the default for both main chat and auxiliary calls. Anthropic-style `cache_control` breakpoints on the persona block when the model supports it.
- **State model:** both lemon and the user are modelled with the same three-layer schema (Big 5 traits + characteristic adaptations + PAD core affect). Tonic state nudges happen pre-reply for both agents in a single merged round-trip; response generation reads from the post-nudge states. See `docs/dyadic_state.md` for the full design.

Everything is synchronous on the critical path; post-gen bookkeeping runs in a daemon thread after the reply is delivered. One process, one conversation at a time. No auth, no multi-user.

**Per-turn LLM budget:**

| call | role | blocks the user? |
|---|---|---|
| `user_read` (STATE_MODEL) | emotion + ToM + user_state_delta + lemon_state_delta | yes |
| `generate_reply` (CHAT_MODEL) | the actual reply, streamed | yes |
| `generate_reply` (retry, conditional) | regenerate on empathy-check failure | yes, rare |
| `bookkeep` (STATE_MODEL) | fact extraction (facts-only) | **no** — backgrounded |

**User-perceived wait = 2 LLM calls.** Total cost = 3 per typical turn (4 when retry fires).

---

## 2. Setup, configuration, environment

### 2.1 Install and run

```bash
pip install -r requirements.txt
echo 'OPENROUTER_API_KEY=sk-or-v1-...' > .env
PYTHONPATH=src python -m app.lem                         # CLI
PYTHONPATH=src python -m app.web                         # web UI on 127.0.0.1:8000
# or:
uvicorn app.web:app --reload --app-dir src               # alternative web launch
pytest                                                   # test suite (after pip install -r requirements-dev.txt)
```

`pytest.ini` sets `pythonpath = src` so the test suite imports `app.*`,
`core.*`, `prompts.*`, etc. without manual path setup.

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
  app/                       entry points + per-turn orchestration
    __init__.py              empty package marker
    lem.py                   CLI REPL entry point (python -m app.lem)
    web.py                   FastAPI app + SSE + introspection endpoints
    pipeline.py              orchestrator: the "empathy pipeline"
    session_context.py       shared CLI+web helpers: initial history, refresh blocks, bookkeeping thread
    commands.py              slash-command registry + ChatContext + dispatcher

  core/                      cross-cutting infrastructure
    __init__.py              empty package marker
    config.py                env, models, paths, knobs, HTTP headers
    logging_setup.py         lemon.* logger tree + payload-safe formatters

  prompts/                   every prompt + every block formatter
    __init__.py              the big one — block formatters, build_user_read_prompt, build_bookkeep_prompt, EMOTION_LABELS, all the *_TAG constants
    persona.py               LEMON_TRAITS (Big 5) + LEMON_ADAPTATIONS (goals/Schwartz-tagged values/concerns/stance)
    prompt_stack.py          replace_system_block + compress_history
    schwartz.py              Schwartz's 10 universal values + alias coercion + entry normalizer

  empathy/                   empathy-pipeline-specific logic
    emotion.py               23-label emotion schema, families, validator
    tom.py                   theory-of-mind schema, validator
    fact_extractor.py        fact-key regex + value-hygiene validator
    empathy_check.py         12-detector regex post-check
    user_read.py             merged pre-gen LLM call (4 sub-objects, including both state deltas)
    post_exchange.py         post-gen LLM call (facts only since stage 2)

  llm/                       raw LLM wire + parsing helpers
    chat.py                  OpenRouter reply call, cache wrap, streaming
    parse_utils.py           shared fence-stripper + recent-msgs prompt formatter

  storage/                   persistence + retrieval
    db.py                    SQLite layer: schema, migrations, CRUD helpers
    memory.py                emotion-tagged message retrieval + <emotional_memory> formatter
    lemon_state.py           lemon's three-layer state: defaults, load/save, validator, legacy migrator
    user_state.py            user's three-layer state: defaults, load/save, validator, apply_delta
    state.py                 DEPRECATED legacy 6-field state (kept as shim for migration path)

  temporal/                  time-context helpers (humanize_age, time_of_day, session_duration_note)

  templates/
    index.html               single-page web UI (vanilla JS, inline CSS)
  static/
    lemon.png                favicon and on-page logo
tests/                       one test file per src/ module + conftest.py
docs/                        architecture, dyadic_state, slash commands, db schema, web ui, empathy research, memory architecture
```

After the 2026-04-27 reorganization, **everything in `src/` lives in a folder**. The three new packages are:

- **`app`** — entry points (`lem`, `web`) and per-turn orchestration (`pipeline`, `session_context`, `commands`). Internal imports use relative form: `from .commands import ChatContext`.
- **`core`** — cross-cutting infrastructure (`config`, `logging_setup`). Imported as `from core import config` or `from core.config import X`.
- **`prompts`** — every prompt and block formatter. The package's `__init__.py` is the formerly-top-level `prompts.py`. Sub-modules: `persona`, `prompt_stack`, `schwartz`.

Dependency direction flows top-to-bottom:

```
core ──► everything
storage/db ──► storage/{lemon_state, user_state, memory, state-shim}, app/commands, app/pipeline, app/session_context
storage/user_state ──► storage/lemon_state (shared validator + apply_delta)
prompts/persona ──► storage/lemon_state
prompts/schwartz ──► storage/user_state
llm/parse_utils ──► empathy/{emotion,tom,fact_extractor}, empathy/{user_read,post_exchange}
llm/chat ──► app/pipeline
empathy/* ──► app/pipeline, app/session_context
prompts ──► app/session_context, app/pipeline
app/commands + app/session_context + app/pipeline ──► app/lem, app/web
```

There are no cycles. Tests are isolated via an autouse `isolated_db()` fixture in `tests/conftest.py`. `pytest.ini` sets `pythonpath = src` so tests import `app.*`, `core.*`, `prompts.*` directly.

---

## 4. Per-turn request lifecycle

Full path a single user message takes. Numbered callouts are the LLM calls.

```
user types a message
  │
  ▼
[refresh base blocks]  session_context.refresh_base_blocks()
  replaces <time_context>, <lemon_state>, <user_facts> at positions
  1/2/3 via prompt_stack.replace_system_block.
  │
  ▼
if is_command(user_input):
    commands.dispatch(user_input, ctx) ──► system bubble; no LLM call; return
  │
  ▼
pipeline.run_empathy_turn(user_msg, base_history, user_state, lemon_state, ...)
  │
  ├── if ENABLE_EMPATHY_PIPELINE is False:
  │     append user_msg + compress_history + (R) generate_reply + return
  │
  ├── recent = recent_messages_for_context(base_history, n=6)
  ├── trace.user_state_before  = user_state  (or DEFAULT_USER_STATE)
  ├── trace.lemon_state_before = lemon_state (or DEFAULT_LEMON_STATE)
  │
  ├── (1) emotion, tom, user_delta, lemon_delta = empathy.user_read.read_user(   [STATE_MODEL]
  │           user_msg, recent, current_user_state, current_lemon_state, model)
  │       one call, four output dicts
  │       phase: "reading you"
  │       internally: _clamp_lemon_delta enforces lemon's tighter PAD bounds and freezes traits/values
  │       user_state_after = apply_user_state_delta(user_state_before, user_delta)
  │       lemon_state_after = apply_lemon_state_delta(lemon_state_before, lemon_delta)
  │       db.save_user_state_snapshot(user_state_after, session_id)
  │       db.save_lemon_state_snapshot(lemon_state_after, session_id)
  │       db.log_message(user_msg, emotion=..., intensity=..., salience=intensity)
  │
  ├── memories = storage.memory.relevant_memories(emotion.primary,           [SQLite only]
  │                  current_session_id, limit=MEMORY_RETRIEVAL_LIMIT)
  │       phase: "remembering"
  │
  ├── history = base_history, then:
  │     - _inject_block(<lemon_state>, format_lemon_state(lemon_state_after))     ← refreshed post-nudge
  │     - _inject_block(<emotional_memory>, format_memory_block(memories))         if memories
  │     - _inject_block(<user_state>,  format_user_state_block(user_state_after))
  │     - _inject_block(<reading>,     format_reading_block(emotion, tom))
  │
  ├── history.append({role: user, content: user_msg})
  ├── history = compress_history(history, keep_recent=8)
  │
  ├── (R) draft = llm.chat.generate_reply(history, model)                     [CHAT_MODEL]
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
  │       (R') second = llm.chat.generate_reply(retry_history)                [CHAT_MODEL]
  │       if second.strip(): final = second; trace.regenerated = True
  │
  └── db.log_message(assistant, final); return (final, trace)
  │
  ▼
caller (web._stream_reply or lem.main):
  - ctx.last_trace = trace
  - ctx.user_state  = trace.user_state_after
  - ctx.lemon_state = trace.lemon_state_after
  - append user_msg + final to ctx.history
  - deliver the reply (SSE token+done, or CLI print)
  - spawn daemon thread:
      (2) new_facts = empathy.post_exchange.bookkeep(                          [STATE_MODEL]
             user_msg, final, existing_facts, recent, model, max_new)
          db.upsert_fact(...) for each new_fact
          (state already updated pre-reply — no state work in this thread)
```

**Total per turn:** 2 LLM calls on the user-facing critical path (user_read + reply), +1 in the background (facts-only bookkeeping), +1 if the post-check fails (retry). With `/empathy off`: 1 main call on the critical path + 1 backgrounded bookkeep.

**State-first ordering:** stage 2 of the dyadic-state architecture moved both agents' tonic-state nudges to *before* the reply call. The reply call sees `<lemon_state>` containing the *post-nudge* state, so reply tone reflects what lemon "feels" in response to what was just said.

---

## 5. What the model sees each turn

The `messages` list passed to OpenRouter for the main chat call:

```
 0. system: <Who you are>...                   persona, ~5KB, wrapped in cache_control
 1. system: <time_context>...                  refreshed each turn
 2. system: <lemon_state>...                   refreshed each turn, then re-injected post-nudge
 3. system: <user_facts>...                    refreshed each turn, skipped if empty
 4. system: <emotional_memory>...              pipeline-injected, skipped if empty
 5. system: <user_state>...                    pipeline-injected (user's tonic state)
 6. system: <reading>...                       pipeline-injected (phasic emotion + ToM)
 7. system: <earlier_conversation>...          only when len(convo) > KEEP_RECENT_TURNS
 ...
    user / assistant / user / ...              last KEEP_RECENT_TURNS turns verbatim
 N. user: <latest message>                     appended by pipeline just before generate
```

Key invariants:

- **Order of injection is deterministic.** `_inject_block` drops any prior block with the same tag, then inserts the new content just after the leading contiguous block of system messages.
- **`time_context` + `lemon_state` + `user_facts` change every turn** and by construction live at positions 1, 2, 3 (managed by `replace_system_block` in `prompt_stack`).
- **Persona block never changes.** That is what makes the cache hit work.
- **Pipeline-scoped blocks (`<lemon_state>` post-nudge, `<emotional_memory>`, `<user_state>`, `<reading>`)** live inside the history temporarily for one call, then are dropped when the pipeline rebuilds next turn.
- **Tonic-then-phasic ordering** for both agents: `<lemon_state>` (lemon's tonic, position 2), then per-turn `<user_state>` (user's tonic) before `<reading>` (per-turn phasic).

### 5.1 Persona block (`prompts.LEMON_PROMPT`)

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

#### `<time_context>` (`prompts.get_time_context`)

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

#### `<lemon_state>` (`prompts.format_lemon_state`)

Rendered from the three-layer dict in `storage.lemon_state.DEFAULT_LEMON_STATE`:

```python
{
    "traits":      {"openness": 0.5, "conscientiousness": -0.2, "extraversion": 0.3,
                    "agreeableness": 0.8, "neuroticism": -0.6},
    "adaptations": {"current_goals": ["be present for the user", "match their energy without forcing it"],
                    "values":        [{"label": "honesty", "schwartz": "universalism"},
                                      {"label": "warmth without performance", "schwartz": "benevolence"},
                                      {"label": "calm", "schwartz": "security"}],
                    "concerns":      [],
                    "relational_stance": "close friend, not assistant"},
    "state":       {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0, "mood_label": "neutral"},
}
```

Block content (compact prose):

```
<lemon_state>
Mood right now: content (pleasure +0.30, arousal +0.00, dominance +0.10)
You are: somewhat openness, slightly low conscientiousness, somewhat extraversion, high agreeableness, low neuroticism.
What you care about doing: be present for the user, match their energy without forcing it.
What you value: honesty, warmth without performance, calm.
Quietly on your mind: nothing in particular.
Stance with this person: close friend, not assistant.
</lemon_state>
```

Traits and values come from `persona.LEMON_TRAITS` / `persona.LEMON_ADAPTATIONS`.
PAD coordinates and `concerns` are nudged by the pre-reply LLM call.
Replaces the legacy `<internal_state>` block.

#### `<user_facts>` (`prompts.format_user_facts`)

Skipped entirely when the `facts` table is empty.

### 5.3 Pipeline-injected blocks

#### `<user_state>` (`prompts.format_user_state_block`)

The user's tonic state, same shape as `<lemon_state>` but rendered with they/them framing:

```
<user_state>
Mood: tired (pleasure -0.20, arousal +0.10, dominance -0.05)
Roughly: high agreeableness, slightly low neuroticism.
On their mind: prep tuesday exam.
Cares about: family, doing well academically.
Worries: feeling unprepared.
How they're showing up: open, slightly tired.
</user_state>
```

Cold-start (default-shaped state) collapses to a one-line low-confidence notice:
`First read of this person — let your reply do the inferring.`

#### `<reading>` (`prompts.format_reading_block`)

Stage 3 unified block — folded the legacy `<user_emotion>` and `<theory_of_mind>` blocks into one:

```
<reading>
Primary feeling: sadness (moderate, intensity 0.52)
Undertones: loneliness, tired
What they probably want: feel heard, not solved
What they're actually feeling: tired and a little embarrassed about the exam
What helps: stay with it, ask one open question if anything
What to avoid: don't jump to advice
</reading>
```

Intensity word ladder: `<0.3 mild`, `<0.6 moderate`, `<0.85 strong`, else `very strong`.
Pairs with `<user_state>` (tonic) — this block carries the *phasic* layer (per-turn reaction + ToM).

#### `<emotional_memory>` (`prompts.format_memory_block`)

```
<emotional_memory>
- yesterday, when feeling sadness: "i just feel flat about this..."
- 4 days ago, when feeling sadness: "everyone's moved on and I'm still stuck"
</emotional_memory>
```

Timestamps are humanized via `temporal.age.humanize_age`: `today`, `yesterday`, `N days ago`, `N weeks ago`, `N months ago`.

---

## 6. Merged pre-gen call — `empathy.user_read.read_user`

One LLM round-trip that returns `(emotion, tom, user_state_delta, lemon_state_delta)`. Stages 1–3 of the dyadic-state architecture progressively expanded the call from 2 sub-objects to 4; round-trip count is unchanged.

- Model: `STATE_MODEL`. `temperature=0.3`, `max_tokens=900` (bumped from 500 to fit four sub-objects).
- Input: the user message + last ~6 non-system turns + current `user_state` + current `lemon_state`.
- Output: a single JSON object with `"emotion"`, `"tom"`, `"user_state_delta"`, `"lemon_state_delta"` sub-dicts.
- Parsing: `llm.parse_utils.strip_json_fences` → `json.loads` → split sub-dicts → run each through its validator. Lemon's delta also goes through `_clamp_lemon_delta` which enforces the asymmetric dynamics (PAD ±0.10 vs user's ±0.15; trait_nudges and value_add forced to empty).
- Fallback: `(DEFAULT_EMOTION, DEFAULT_TOM, zero_delta, zero_delta)` on any failure. A bad LLM response can never poison either persisted state.
- Side effects:
  - The pipeline applies both deltas via `apply_delta` and persists both via `save_user_state` / `save_lemon_state` *before* the reply call, so reply generation reads from the post-nudge states.
  - The user message row in `messages` is written with `emotion`, `intensity`, and `salience=intensity` populated by the pipeline after this call returns.

Validators:
- `empathy.emotion._validate` — emotion schema (label whitelist, intensity clamp, undertones cap)
- `empathy.tom._validate` — ToM schema (string coercion, null fallbacks)
- `storage.user_state.validate_delta` — both deltas, with magnitude caps
- `empathy.user_read._clamp_lemon_delta` — extra lemon-side asymmetric clamp on top of `validate_delta`

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

## 9. Post-gen call — `empathy.post_exchange.bookkeep` (facts only)

One LLM round-trip after the reply is delivered. Returns `new_facts` (a dict). Stage 2 of the dyadic-state architecture moved the state nudge half pre-reply (into the merged `user_read` call), so this call is now facts-only.

- Model: `STATE_MODEL`. `temperature=0.2`, `max_tokens=350` (tightened from 500 since output shrunk).
- Input: the user message + bot reply + existing facts dict + last ~6 turns.
- Output: a single JSON object. Accepts either `{"facts": {...}}` or a flat `{...}` for compatibility.
- Parsing: strip fences → validate via `empathy.fact_extractor._validate`.
- Runs in a daemon thread from both `web.py::_stream_reply` and `lem.py::main`. See `session_context.run_bookkeeping`.
- Failure: returns `{}` and logs the error. Never blocks or breaks the chat.

**The user never waits for this call.** The reply is delivered first; fact extraction follows. On clean CLI exit, `lem.py` joins the final bookkeeping thread (timeout 10s) so the last fact upserts aren't lost. State writes happen pre-reply now, so they're already on disk before this thread runs.

---

## 10. Web app — `web.py`

### 10.1 Startup

At import time the module:

1. Creates a `FastAPI` instance and a `threading.Lock()` for serialising access to the shared ChatContext.
2. Calls `db.start_session()` once, records the resulting `session_id` globally.
3. Loads lemon's latest tonic state via `storage.lemon_state.fresh_lemon_session_state()` (latest snapshot + session-start PAD re-peg + persona-baseline relational_stance reset; concerns and goals carry over). On a fresh install with no `lemon_state_snapshots` rows but an old `state_snapshots` row, `migrate_legacy_state` auto-converts the legacy 6-field shape.
4. Loads the user's latest tonic state via `storage.user_state.fresh_user_session_state()` — no session-start overrides on the user side; whatever they were carrying carries over.
5. Builds an initial history via `session_context.initial_history` (persona + time_context + lemon_state + optional facts block).
6. Picks a random opener from `prompts.LEMON_OPENERS`, appends it to `ctx.history` and logs it to `messages`.
7. Reads `templates/index.html` into `_INDEX_HTML` once (not on every `GET /`).

So the first-paint UI already has one assistant bubble in `/history` before the user sends anything.

### 10.2 Endpoints

| method | path           | returns |
|---|---|---|
| GET  | `/`            | `templates/index.html` (cached at startup) |
| POST | `/chat`        | `text/event-stream` (SSE, see §10.3) |
| POST | `/command`     | `{"output": "...", "exit": bool}` |
| GET  | `/state`       | lemon's three-layer tonic state dict |
| GET  | `/user_state`  | user's three-layer tonic state dict (same schema) |
| GET  | `/facts`       | `{key: value, ...}` from `facts` table |
| GET  | `/sessions`    | last 20 sessions with msg counts |
| GET  | `/history`     | non-system messages in the current in-memory session |
| GET  | `/trace`       | the last `PipelineTrace` serialised (both state trajectories included) |
| GET  | `/docs`        | FastAPI's auto-generated OpenAPI explorer |

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

- `bubble(role, text)` renders one chat bubble; classes `you`, `lemon`, or `system`. System bubbles include a collapse/expand toggle in the upper-right corner.
- `typingIndicator()` returns `{el, setPhase}`; `setPhase(p)` picks a random phrase from a flat `PHASE_PHRASES` list (~45 ice-cream-themed phrases) and shows `"lemon is <phrase>"` with a cross-fade animation.
- `streamChat(message)` POSTs to `/chat` and consumes the SSE stream. On the first `token` event, removes the typing indicator and opens a new lemon bubble. Splits on `\n\n` for iMessage-style multi-bubble replies.
- `formatStateBlock(state, opts)` renders a three-layer state object as compact prose (mood + PAD coords, optional Big 5 trait descriptors, goals, values, concerns, stance). Used by both sidebar state sections.
- `loadState` / `loadUserState` / `loadFacts` / `loadSessions` populate the sidebar. All four refresh after every chat turn AND every slash command.
- `runCommand(text)` POSTs to `/command` and renders the response in a system bubble.

Light/dark theme toggle via CSS custom properties (`:root[data-theme="dark"]`) with the saved preference applied before body paint to avoid a flash. Default is light.

### 10.5 Concurrency model

`threading.Lock()` wraps every read/write of the shared `ChatContext` on both sides:
- The SSE generator acquires the lock to snapshot `base_history` and read `ctx.user_state` / `ctx.lemon_state`, then releases it during `run_empathy_turn`.
- After the pipeline returns, the lock is reacquired to write back the new state values from the trace.
- The bookkeeping daemon thread acquires the same lock before calling `db.upsert_fact` (state writes happen pre-reply now, inside the pipeline's read step, so the bookkeeping thread no longer touches state).

Because the web server is meant for the user themselves (localhost), there is no per-user isolation. Running two browser tabs against the same server interleaves their messages into one conversation.

---

## 11. CLI — `lem.py`

REPL loop:

```
1. db.start_session(), fresh_lemon_session_state(), fresh_user_session_state(), build initial history
2. print a random opener, log it
3. while not exit_requested:
     user_input = input("you: ")
     if is_command: dispatch; continue
     base_history = session_context.refresh_base_blocks(history, lemon_state, ...)
     reply, trace = run_empathy_turn(
         user_input, base_history,
         user_state=ctx.user_state,
         lemon_state=ctx.lemon_state,
         ...
     )
     ctx.last_trace = trace
     ctx.user_state  = trace.user_state_after
     ctx.lemon_state = trace.lemon_state_after
     append user + reply to ctx.history
     print(f"lemon: {reply}\n")
     spawn daemon thread: session_context.run_bookkeeping(...)   # facts only
4. finally:
     last_bg.join(timeout=10)   # wait for last fact extraction
     save_lemon_state + end_session(session_id)
```

Phase updates are printed inline as `  · reading you...`, `  · replying...` etc, via an `on_phase` callback.

---

## 12. Slash commands — `commands.py`

Every command is a function decorated with `@command("name", "help text")`. The decorator appends to the module-level `_REGISTRY`. `dispatch(text, ctx)` parses `/name arg...` and invokes the handler. Both the CLI and the `/command` endpoint call the same dispatcher; adding a new command works in both frontends with zero client-side changes.

Current commands:

| command | what it does |
|---|---|
| `/help` | list all commands |
| `/state` | render `ctx.lemon_state` (mood, PAD, traits, adaptations) |
| `/user_state` | render `ctx.user_state` (same shape as lemon's) |
| `/reset` | reset lemon's state to `DEFAULT_LEMON_STATE`, save snapshot |
| `/facts` | list stored facts |
| `/remember key=value` | `db.upsert_fact(key, value, source_session_id=...)` |
| `/forget key` | `db.delete_fact(key)` |
| `/history [n]` | last `n*2` messages from `ctx.history` (non-system) |
| `/rewind` | pop the last two non-system messages from `ctx.history` |
| `/clear` | drop the in-memory chat history this session (db untouched) |
| `/export` | dump this session's chat as plain text |
| `/model name` | set `ctx.chat_model = name` for this session only |
| `/sessions` | `db.list_sessions(limit=10)` with msg counts |
| `/search query` | FTS5 search over `messages.content` |
| `/recall emotion` | past user messages tagged with the given emotion |
| `/stats` | message / session / fact counts |
| `/config` | current behaviour flags (model, empathy, auto-facts, cache, etc.) |
| `/empathy [on\|off]` | mutate `config.ENABLE_EMPATHY_PIPELINE` |
| `/autofacts [on\|off]` | mutate `config.ENABLE_AUTO_FACTS` |
| `/cache [on\|off]` | mutate `config.ENABLE_PROMPT_CACHE` |
| `/why` | render `ctx.last_trace` (both state trajectories included) |
| `/quit` / `/exit` | set `ctx.exit_requested = True` (two decorators, one handler) |

`/why` renders the same data that `/trace` returns as JSON, but pretty-printed.

---

## 13. State layer

Four kinds of persistent state, two of them three-layer dyadic objects:

### 13.1 Lemon state (`storage/lemon_state.py`)

Three-layer dict, same schema as the user side. Rendered into `<lemon_state>` every turn. Nudged pre-reply by the merged `user_read` call (with asymmetric clamping in `_clamp_lemon_delta`). Persisted as a row in `lemon_state_snapshots` whenever it changes.

```python
{
    "traits":      {"openness": float, "conscientiousness": float, "extraversion": float,
                    "agreeableness": float, "neuroticism": float},   # each in [-1, +1]
    "adaptations": {"current_goals":     list[str],
                    "values":            list[dict],   # tagged: {label, schwartz: str | None}
                    "concerns":          list[str],
                    "relational_stance": str | None},
    "state":       {"pleasure": float, "arousal": float, "dominance": float,
                    "mood_label": str},   # PAD in [-1, +1]; mood_label from a small whitelist
}
```

Traits are hardcoded from `persona.LEMON_TRAITS` and never drift (validator-enforced via `_clamp_lemon_delta`'s `trait_nudges = {}`). Adaptations are seeded from `persona.LEMON_ADAPTATIONS`; `concerns` may grow during a session. `fresh_lemon_session_state` re-pegs PAD to `LEMON_SESSION_START_STATE` (`pleasure=+0.30, arousal=+0.00, dominance=+0.10, mood_label="content"`) and resets `relational_stance` to the persona baseline; concerns and goals carry over for cross-session continuity.

### 13.2 User state (`storage/user_state.py`)

Same shape as lemon's. Traits inferred slowly (per-turn cap effectively freezes them in stage 1). Adaptations grow as the user shares goals/values/concerns. PAD updates per turn. Persisted as a row in `user_state_snapshots`.

`fresh_user_session_state` has **no overrides** — whatever the user was carrying when the last session ended carries into the next one.

### 13.3 User facts

Key-value table. Upsert via `/remember` or `post_exchange.bookkeep`, delete via `/forget`. Rendered into the `<user_facts>` block every turn when non-empty. Facts persist across sessions.

### 13.4 Emotion-tagged messages

Every user message row gets `emotion`, `intensity`, and `salience` populated by the pipeline at log time (in `pipeline.run_empathy_turn`, right after the `user_read` call). Assistant rows leave those three columns NULL. `storage.memory.relevant_memories()` reads these rows via `db.find_messages_by_emotion` and `db.find_messages_by_fts`, filtered to user messages in other sessions.

### 13.5 Legacy 6-field state (deprecated)

`storage/state.py` and the `state_snapshots` table remain as a deprecated shim. New writes go to `lemon_state_snapshots`. On startup, `load_lemon_state` checks for a `lemon_state_snapshots` row first; if absent but a legacy `state_snapshots` row exists, `migrate_legacy_state` converts the 6-field shape into the three-layer one (mood + energy → PAD; disposition → relational_stance; emotional_thread → first concern; recent_activity dropped) and persists it as the first lemon_state snapshot.

---

## 14. Database layer — `storage/db.py`

See `db_schema.md` for column-by-column details.

- Six tables: `sessions`, `messages`, `state_snapshots` (legacy archive), `lemon_state_snapshots`, `user_state_snapshots`, `facts`. Plus `messages_fts` (FTS5 virtual table) and `schema_version` for migration bookkeeping.
- Idempotent schema: `CREATE TABLE IF NOT EXISTS` on every connect.
- Migrations are a list of `(version, [SQL statements])`. On connect, any `version > current` is applied. "duplicate column name" errors are swallowed because fresh databases already have the column from the base SCHEMA.
- Schema version is currently 3. Migrations: (1) ALTER messages add emotion/intensity/salience; (2) CREATE user_state_snapshots; (3) CREATE lemon_state_snapshots.
- Every helper opens a short-lived connection via `@contextmanager connect()`.
- Every helper accepts an optional `path=` argument so tests can point at a per-test tmp file.

Access pattern (user-facing helpers):

```
start_session() → int
end_session(sid)
list_sessions(limit)

log_message(sid, role, content, emotion=, intensity=, salience=)
find_messages_by_emotion(emotion, exclude_session_id=, limit=)
find_messages_by_fts(fts_query, exclude_session_id=, candidate_pool=)
find_recent_user_messages(exclude_session_id=, limit=)

latest_state()              → dict | None    # legacy 6-field, for the migration shim
save_state_snapshot(state, session_id=)      # not called by current code (archive only)

latest_lemon_state()        → dict | None    # three-layer schema
save_lemon_state_snapshot(state, session_id=)

latest_user_state()         → dict | None    # three-layer schema
save_user_state_snapshot(state, session_id=)

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
| `user_read` (emotion + ToM + user_delta + lemon_delta) | every turn | ~$0.0002–0.0003 |
| main chat (draft) | every turn | varies with context size; persona cached after turn 1 |
| main chat (retry) | only on post-check fail | same as draft |
| `bookkeep` (facts only) | every turn, backgrounded | ~$0.0001 |

Per-turn ceiling with pipeline on: 3 Haiku calls (2 user-blocking). With `/empathy off`: 1 main call user-blocking + 1 backgrounded bookkeep.

Stages 2+3 of the dyadic-state architecture kept the round-trip count constant: the new lemon_state_delta rides on the existing user_read call (longer JSON, same call), and bookkeep got cheaper because its state half is gone.

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
emotion:             dict | None    # phasic emotion classifier output
memories:            list[dict]     # raw message rows retrieved
tom:                 dict | None    # theory-of-mind output
draft:               str | None     # initial generation (before retry)
check:               CheckResult | None
regenerated:         bool
final:               str | None     # what lemon actually sent
pipeline_used:       bool           # False when ENABLE_EMPATHY_PIPELINE was off
facts_extracted:     dict           # populated by the bookkeeping thread after the reply ships

# Dyadic-state stage 1 — user_state trajectory
user_state_before:   dict | None
user_state_after:    dict | None
user_state_delta:    dict | None

# Dyadic-state stages 2+3 — lemon_state trajectory
lemon_state_before:  dict | None
lemon_state_after:   dict | None
lemon_state_delta:   dict | None
```

The trace is attached to `ctx.last_trace` after every reply and persists only in memory. `facts_extracted` appears after a brief delay — bookkeeping runs post-reply. State trajectories are populated *before* the reply ships (state-first, response-second ordering).

---

## 19. Extending the system

### 19.1 Add a slash command

Drop this into `src/commands.py`:

```python
@command("mood", "force lemon's mood label: /mood happy")
def _mood(ctx: ChatContext, args: str) -> CommandResult:
    from storage.user_state import MOOD_LABELS
    from storage.lemon_state import save_lemon_state
    new = args.strip()
    if not new:
        return CommandResult(f"current mood: {ctx.lemon_state['state']['mood_label']}")
    if new not in MOOD_LABELS:
        return CommandResult(f"unknown mood {new!r}. valid: {', '.join(MOOD_LABELS)}")
    ctx.lemon_state["state"]["mood_label"] = new
    save_lemon_state(ctx.lemon_state, session_id=ctx.session_id)
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
- **Fact-extraction lag.** New facts appear in `/facts` a few seconds after the reply (post-gen thread is async). `/trace.facts_extracted` is populated after that thread finishes. State updates do *not* lag — they happen pre-reply now.
- **SSE not resumable.** If the browser tab closes mid-stream the reply was still generated and logged; the user just doesn't see the replay.
- **Post-check is regex, not semantic.** Paraphrases of minimizing or validation-cascade phrases will still slip through. See `empathy_research.md` §2 for semantic alternatives.
- **Emotion classifier labels are coarse.** 23-label taxonomy drawn from GoEmotions with the self-conscious cluster (Tracy & Robins) and `relief`. See `docs/dyadic_state.md` §6.3 for the rationale.
- **User trait inference is essentially frozen** in stage 1. The `_TRAIT_NUDGE_CAP = 0.02` per turn means traits drift very slowly. Long-cadence trait re-estimation is future work.
- **ToM pass doesn't see the memory block.** It consumes the emotion read + last ~6 turns + both states.
- **Mood label can drift from PAD coords** in pathological LLM outputs — the validator enforces independent vocabularies. Future work could enforce mood_label↔PAD consistency.
- **Some legacy test files are stale.** `test_emotion.py`, `test_fact_extractor.py`, `test_tom.py`, `test_chat.py`, plus parts of `test_db.py` reference pre-refactor symbols (`classify_emotion`, `humanize_delay`, etc.). They collect-error or fail with AttributeError rather than run. The current dyadic-state flow is covered by `test_user_state.py` (29 tests), `test_lemon_state.py` (22 tests), `test_pipeline.py` (with both state-trajectory tests), `test_state.py` (legacy shim), and `test_empathy_check.py` (46 tests).

---

## 21. Quick reference

**Start a chat from scratch:**

```bash
PYTHONPATH=src python -m app.lem    # CLI
PYTHONPATH=src python -m app.web    # web, open http://127.0.0.1:8000
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
LEMON_EMPATHY=0 PYTHONPATH=src python -m app.web
```

**Switch model for one session:**

```
/model anthropic/claude-sonnet-4.6
```

---

## 22. File-by-file cheat sheet

| file | purpose |
|---|---|
| `core/config.py` | env vars, model IDs, knobs, HTTP headers |
| `core/logging_setup.py` | `lemon.*` logger tree, `setup_logging`, `get_logger`, payload-safe formatters |
| `app/pipeline.py` | orchestrator: `read_user (→ both states updated) → memory → inject blocks → draft → check → regen-once` |
| `app/session_context.py` | `initial_history`, `refresh_base_blocks`, `run_bookkeeping` — shared CLI+web |
| `app/commands.py` | slash-command registry + 22 built-ins; `ChatContext` (history, lemon_state, user_state, etc.) |
| `app/lem.py` | CLI REPL (`python -m app.lem`) |
| `app/web.py` | FastAPI app + SSE + introspection (`python -m app.web` or `uvicorn app.web:app --app-dir src`) |
| `prompts/__init__.py` | single source of truth for every prompt + every block formatter |
| `prompts/persona.py` | `LEMON_TRAITS` (Big 5) + `LEMON_ADAPTATIONS` (goals/Schwartz-tagged values/concerns/stance) |
| `prompts/schwartz.py` | Schwartz's 10 universal values, descriptions, `coerce_schwartz`, `normalize_value_entry` |
| `prompts/prompt_stack.py` | `replace_system_block` + `compress_history` |
| `empathy/emotion.py` | 23-label emotion schema, family map, `_validate` |
| `empathy/tom.py` | ToM schema, `_validate` |
| `empathy/fact_extractor.py` | fact-key regex + `_validate` |
| `empathy/empathy_check.py` | 12-detector post-check |
| `empathy/user_read.py` | merged pre-gen LLM call (4 sub-objects, including both state deltas) |
| `empathy/post_exchange.py` | post-gen LLM call (facts only) |
| `llm/chat.py` | OpenRouter reply call, cache wrap, streaming |
| `llm/parse_utils.py` | shared fence-stripper + recent-msgs formatter |
| `storage/db.py` | schema, migrations, CRUD helpers |
| `storage/memory.py` | composite-scored retrieval + `<emotional_memory>` formatter |
| `storage/lemon_state.py` | lemon's three-layer state: defaults, load/save, validator, `migrate_legacy_state` |
| `storage/user_state.py` | user's three-layer state: defaults, load/save, validator, `apply_delta` |
| `storage/state.py` | DEPRECATED legacy 6-field state shim, kept for migration path |
| `templates/index.html` | single-page web UI (sidebar shows lemon's state + user's state separately) |
| `static/lemon.png` | favicon and on-page logo |
