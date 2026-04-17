# SQLite schema

One file (`.lemon.db` by default — override with `LEMON_DB`). Schema lives in `src/db.py` as a single `CREATE TABLE IF NOT EXISTS` script that runs on every connect, so the file self-migrates.

## Tables

### `sessions`

| column      | type    | notes                                              |
| ----------- | ------- | -------------------------------------------------- |
| `id`        | INTEGER | primary key                                        |
| `started_at`| TEXT    | ISO timestamp, set on `start_session()`            |
| `ended_at`  | TEXT    | ISO timestamp, set on `end_session()` (nullable)   |

A row exists per process invocation of `lem.py` or per server start of `web.py`.

### `messages`

| column       | type    | notes                                       |
| ------------ | ------- | ------------------------------------------- |
| `id`         | INTEGER | primary key                                 |
| `session_id` | INTEGER | FK → sessions(id), `ON DELETE CASCADE`      |
| `role`       | TEXT    | `system` / `user` / `assistant`             |
| `content`    | TEXT    | the message body                            |
| `created_at` | TEXT    | ISO timestamp                               |

Index: `idx_messages_session(session_id)`.

We log `user` and `assistant` messages in the chat loop. System blocks (persona, time, internal state, facts) are *not* logged here — they're regenerated each turn anyway.

### `state_snapshots`

| column       | type    | notes                                                 |
| ------------ | ------- | ----------------------------------------------------- |
| `id`         | INTEGER | primary key                                           |
| `session_id` | INTEGER | FK → sessions(id), `ON DELETE SET NULL` (nullable)    |
| `created_at` | TEXT    | ISO timestamp                                         |
| `state_json` | TEXT    | the full internal-state dict serialized as JSON       |

Index: `idx_state_session(session_id)`.

A new row is appended every time `save_state()` is called — i.e. after every state-updater run, after `/reset`, and on session shutdown. `latest_state()` returns the most recent row's `state_json` parsed.

### `facts`

| column              | type    | notes                                              |
| ------------------- | ------- | -------------------------------------------------- |
| `key`               | TEXT    | primary key                                        |
| `value`             | TEXT    |                                                    |
| `source_session_id` | INTEGER | FK → sessions(id), `ON DELETE SET NULL` (nullable) |
| `created_at`        | TEXT    | ISO timestamp                                      |
| `updated_at`        | TEXT    | ISO timestamp                                      |

Upsert semantics: `INSERT ... ON CONFLICT(key) DO UPDATE`. Adding the same key twice updates the value, doesn't duplicate.

## Access pattern

Everything goes through helper functions in `db.py`:

```python
sid = db.start_session()                              # → int
db.log_message(sid, "user", "hi")
db.log_message(sid, "assistant", "hey")
db.save_state_snapshot({...}, session_id=sid)
db.upsert_fact("city", "Bangalore", source_session_id=sid)
db.end_session(sid)

db.latest_state()              # → dict | None
db.list_sessions(limit=20)     # → list[dict]
db.get_facts()                 # → dict[str, str]
db.session_messages(sid)       # → list[dict]
```

Every function takes an optional `path` argument that defaults to `config.DB_PATH` resolved at call time. Tests inject a per-test `tmp_path` via the autouse `isolated_db` fixture in `conftest.py`.

## Notes on durability

- Each helper opens its own short-lived connection inside a context manager and commits on exit. There is no long-lived connection — fine for a single-user CLI/web app, would need rethinking for high-throughput.
- Foreign keys are enabled (`PRAGMA foreign_keys = ON`). Inserting a `state_snapshot` with a `session_id` that doesn't exist will fail.
- WAL mode is *not* enabled. If you ever want to read the db while a long chat is running, add `PRAGMA journal_mode = WAL` in `connect()`.

## Inspecting the db

```bash
sqlite3 .lemon.db
sqlite> .tables
sqlite> SELECT id, started_at, ended_at FROM sessions ORDER BY id DESC LIMIT 5;
sqlite> SELECT role, content FROM messages WHERE session_id = 7 ORDER BY id;
sqlite> SELECT key, value FROM facts;
sqlite> SELECT created_at, state_json FROM state_snapshots ORDER BY id DESC LIMIT 3;
```
