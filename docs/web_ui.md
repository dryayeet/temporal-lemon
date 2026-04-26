# Web UI

`src/web.py` is a small FastAPI app that wraps the same backend the CLI uses. The frontend is a single-file `src/templates/index.html` — vanilla JS, no build step.

## Run

```bash
python src/web.py
# or
uvicorn web:app --reload --app-dir src
```

Then open `http://127.0.0.1:8000`.

The app is single-process, single-user. There is no authentication. Don't expose it past localhost without a reverse proxy and an auth layer.

## Endpoints

| method | path           | purpose                                                              |
| ------ | -------------- | -------------------------------------------------------------------- |
| GET    | `/`            | serves `templates/index.html` (cached at startup)                    |
| POST   | `/chat`        | streams a chat reply. Body: `{"message": "..."}`. Returns SSE.       |
| POST   | `/command`     | runs a slash command. Body: `{"text": "/help"}`. Returns JSON.       |
| GET    | `/state`       | lemon's three-layer tonic state (traits / adaptations / PAD)         |
| GET    | `/user_state`  | the user's inferred three-layer tonic state (same schema)            |
| GET    | `/facts`       | stored user facts as JSON                                            |
| GET    | `/sessions`    | recent sessions (id, started_at, ended_at, msg_count)                |
| GET    | `/history`     | non-system messages from the current in-memory session               |
| GET    | `/trace`       | last `PipelineTrace` as JSON (emotion, ToM, memories, check, facts, both state trajectories) |
| GET    | `/docs`        | FastAPI's auto-generated OpenAPI explorer                            |

`/state` and `/user_state` return the same shape — three layers (`traits`,
`adaptations`, `state`) per `docs/dyadic_state.md` §6. The asymmetry between
the two agents lives in the *dynamics* (lemon's traits hardcoded in
`src/persona.py`, lemon's PAD damped harder), not the structure.

## SSE protocol for `/chat`

Each event line is `data: <json>\n\n`. The JSON always has `event` and `data`:

```jsonc
{ "event": "phase", "data": "reading you" }   // pipeline-phase label for the typing indicator
{ "event": "phase", "data": "remembering" }
{ "event": "phase", "data": "replying" }
{ "event": "token", "data": "hey you good?" } // the full reply, one chunk
{ "event": "done",  "data": "hey you good?" } // final aggregated reply
{ "event": "error", "data": "HTTP 401: ..." } // sent if the upstream fails
```

Phase events are buffered server-side while the pipeline runs and flushed just before the token chunk. Consecutive duplicates are deduped. The `token` event carries the full reply body (we buffer server-side because the empathy post-check needs a complete draft before deciding whether to regenerate).

After `done` ships, a daemon thread runs the post-gen bookkeeping call (facts-only since stage 2 of the dyadic-state architecture). `/facts` and `/trace.facts_extracted` update within a few seconds. `/state` and `/user_state` are already current at that point — the state nudges happen *pre-reply* now, inside the merged user_read pass.

## Frontend layout

- **Sidebar** — four sections, each with a refresh button: lemon's state, user's state, facts, recent sessions. Both state sections render a compact prose summary of the underlying three-layer object (mood + PAD coordinates, then goals / values / concerns / stance; user's section also surfaces inferred Big 5 trait descriptors). All four auto-refresh after each completed reply and after every slash command.
- **Header** — lemon logo + `/help` hint + light/dark toggle.
- **Messages** — chat bubbles. User on the right, lemon on the left, system (slash command output, errors) centered in monospace. System bubbles have a collapse/expand toggle in the upper-right corner.
- **Composer** — single-line input. Enter sends, Shift+Enter inserts a newline. Anything starting with `/` is routed to `/command` instead of `/chat`.

The CSS supports both light (default lemon-sorbet palette) and dark mode via a manual toggle in the header (preference stored in `localStorage`).

## Multi-user (not implemented)

The current design assumes a single user. To go multi-user you would need:

1. A `users` table in `storage/db.py`, with a `user_id` foreign key on `sessions`, `lemon_state_snapshots`, `user_state_snapshots`, `facts`, and `messages`.
2. Auth — even something basic like a signed cookie or the OpenRouter API key proxied through.
3. A per-user `ChatContext` map in `web.py`, keyed by user id, instead of the single module-level `_ctx`.

Treat the present setup as a personal-use demo, not production.
