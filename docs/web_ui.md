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

| method | path         | purpose                                                                |
| ------ | ------------ | ---------------------------------------------------------------------- |
| GET    | `/`          | serves `templates/index.html`                                          |
| POST   | `/chat`      | streams a chat reply. Body: `{"message": "..."}`. Returns SSE.         |
| POST   | `/command`   | runs a slash command. Body: `{"text": "/help"}`. Returns JSON.         |
| GET    | `/state`     | current internal state as JSON                                         |
| GET    | `/facts`     | stored user facts as JSON                                              |
| GET    | `/sessions`  | recent sessions (id, started_at, ended_at, msg_count)                  |
| GET    | `/history`   | non-system messages from the current in-memory session                 |
| GET    | `/docs`      | FastAPI's auto-generated OpenAPI explorer                              |

## SSE protocol for `/chat`

Each event line is `data: <json>\n\n`. The JSON has two fields:

```jsonc
{ "event": "token", "data": "hey " }   // a streamed text delta
{ "event": "token", "data": "you" }
{ "event": "token", "data": " good?" }
{ "event": "done",  "data": "hey you good?" }   // final aggregated reply
{ "event": "error", "data": "HTTP 401: ..." }   // sent if the upstream fails
```

The frontend appends `token` events into a "lemon is typing" bubble and updates the sidebar state on `done`. Errors collapse the partial bubble and render a system message.

Tokens are paced server-side (`humanize_delay`) so the perceived typing speed reflects lemon's `energy` state — a tired lemon types slower than an upbeat one.

## Frontend layout

- **Sidebar** — internal state, facts, recent sessions. Each section has a refresh button. State auto-refreshes after each completed reply.
- **Header** — `🍋 lemon` and a hint about `/help`.
- **Messages** — chat bubbles. User on the right, lemon on the left, system (slash command output, errors) centered in monospace.
- **Composer** — single-line input. Enter sends, Shift+Enter inserts a newline. Anything starting with `/` is routed to `/command` instead of `/chat`.

The CSS auto-switches between light and dark themes via `prefers-color-scheme`.

## Multi-user (not implemented)

The current design assumes a single user. To go multi-user you would need:

1. A `users` table in `db.py`, with a `user_id` foreign key on `sessions`, `state_snapshots`, `facts`, and `messages`.
2. Auth — even something basic like a signed cookie or the OpenRouter API key proxied through.
3. A per-user `ChatContext` map in `web.py`, keyed by user id, instead of the single module-level `_ctx`.

Treat the present setup as a personal-use demo, not production.
