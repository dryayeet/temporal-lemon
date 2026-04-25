"""SQLite-backed persistence for lemon.

Holds long-term memory: every session, every message, the internal-state
trajectory, user facts, and per-turn emotion classifications used by the
empathy pipeline.

Schema is created idempotently on connect; column additions for existing
databases run through a tiny version-bumped migrations table.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import config

# Fresh databases get this. Existing databases get the same shape via the
# migrations list below, which adds columns one version at a time.

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS sessions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   TEXT NOT NULL,
    ended_at     TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role         TEXT NOT NULL,        -- system | user | assistant
    content      TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    emotion      TEXT,                 -- detected primary emotion (user msgs)
    intensity    REAL,                 -- 0.0–1.0
    salience     REAL                  -- 0.0–1.0 retrieval weight
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_emotion ON messages(emotion);

CREATE TABLE IF NOT EXISTS state_snapshots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    created_at   TEXT NOT NULL,
    state_json   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_state_session ON state_snapshots(session_id);

CREATE TABLE IF NOT EXISTS facts (
    key          TEXT PRIMARY KEY,
    value        TEXT NOT NULL,
    source_session_id INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
"""

# (target_version, [statements to bump TO that version]).
# "duplicate column name" errors are tolerated because SCHEMA above already
# adds these on a freshly-created database — only existing dbs need the ALTER.
MIGRATIONS: list[tuple[int, list[str]]] = [
    (1, [
        "ALTER TABLE messages ADD COLUMN emotion TEXT",
        "ALTER TABLE messages ADD COLUMN intensity REAL",
        "ALTER TABLE messages ADD COLUMN salience REAL",
    ]),
]

LATEST_VERSION = max((v for v, _ in MIGRATIONS), default=0)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _resolve(path: Optional[Path]) -> Path:
    return path if path is not None else config.DB_PATH


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring an existing db up to LATEST_VERSION. No-op on a fresh db."""
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    current = row[0] if row and row[0] is not None else 0

    for target, statements in MIGRATIONS:
        if target <= current:
            continue
        for stmt in statements:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise
        conn.execute(
            "INSERT OR IGNORE INTO schema_version (version) VALUES (?)", (target,)
        )


@contextmanager
def connect(path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    """Yield a connection with foreign keys enabled, schema in place, and migrations applied."""
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.execute(
        "INSERT OR IGNORE INTO schema_version (version) VALUES (?)", (LATEST_VERSION,)
    )
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------- sessions ----------

def start_session(path: Optional[Path] = None) -> int:
    with connect(path) as c:
        cur = c.execute(
            "INSERT INTO sessions (started_at) VALUES (?)", (_now(),)
        )
        return cur.lastrowid


def end_session(session_id: int, path: Optional[Path] = None) -> None:
    with connect(path) as c:
        c.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            (_now(), session_id),
        )


def list_sessions(limit: int = 20, path: Optional[Path] = None) -> list[dict]:
    with connect(path) as c:
        rows = c.execute(
            "SELECT id, started_at, ended_at, "
            "       (SELECT COUNT(*) FROM messages WHERE session_id = sessions.id) AS msg_count "
            "FROM sessions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------- messages ----------

def log_message(
    session_id: int,
    role: str,
    content: str,
    emotion: Optional[str] = None,
    intensity: Optional[float] = None,
    salience: Optional[float] = None,
    path: Optional[Path] = None,
) -> int:
    """Insert a message row. Returns the new message id."""
    with connect(path) as c:
        cur = c.execute(
            "INSERT INTO messages "
            "(session_id, role, content, created_at, emotion, intensity, salience) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, role, content, _now(), emotion, intensity, salience),
        )
        return cur.lastrowid


def find_messages_by_emotion(
    emotion: str,
    exclude_session_id: Optional[int] = None,
    limit: int = 5,
    path: Optional[Path] = None,
) -> list[dict]:
    """Return user messages tagged with `emotion` from past sessions, newest first."""
    sql = (
        "SELECT id, session_id, role, content, created_at, emotion, intensity, salience "
        "FROM messages WHERE role = 'user' AND emotion = ? "
    )
    params: list = [emotion]
    if exclude_session_id is not None:
        sql += "AND session_id != ? "
        params.append(exclude_session_id)
    sql += "ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with connect(path) as c:
        rows = c.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


# ---------- state snapshots ----------

def latest_state(path: Optional[Path] = None) -> Optional[dict]:
    with connect(path) as c:
        row = c.execute(
            "SELECT state_json FROM state_snapshots ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return json.loads(row["state_json"]) if row else None


def save_state_snapshot(state: dict, session_id: Optional[int] = None,
                        path: Optional[Path] = None) -> None:
    with connect(path) as c:
        c.execute(
            "INSERT INTO state_snapshots (session_id, created_at, state_json) "
            "VALUES (?, ?, ?)",
            (session_id, _now(), json.dumps(state)),
        )


# ---------- facts ----------

def upsert_fact(key: str, value: str, source_session_id: Optional[int] = None,
                path: Optional[Path] = None) -> None:
    now = _now()
    with connect(path) as c:
        c.execute(
            """
            INSERT INTO facts (key, value, source_session_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                source_session_id = excluded.source_session_id,
                updated_at = excluded.updated_at
            """,
            (key, value, source_session_id, now, now),
        )


def get_facts(path: Optional[Path] = None) -> dict[str, str]:
    with connect(path) as c:
        rows = c.execute("SELECT key, value FROM facts ORDER BY key").fetchall()
        return {r["key"]: r["value"] for r in rows}


def delete_fact(key: str, path: Optional[Path] = None) -> bool:
    with connect(path) as c:
        cur = c.execute("DELETE FROM facts WHERE key = ?", (key,))
        return cur.rowcount > 0
