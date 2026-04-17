"""SQLite-backed persistence for lemon.

Holds long-term memory: every session, every message, the internal-state
trajectory, and any user facts the bot has extracted. One file, one
schema, idempotent migrations on connect.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import config

SCHEMA = """
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
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

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


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _resolve(path: Optional[Path]) -> Path:
    """Resolve to the requested path, or fall back to the live config setting."""
    return path if path is not None else config.DB_PATH


@contextmanager
def connect(path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    """Yield a connection with foreign keys enabled and the schema in place."""
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
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

def log_message(session_id: int, role: str, content: str, path: Optional[Path] = None) -> None:
    with connect(path) as c:
        c.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, _now()),
        )


def session_messages(session_id: int, path: Optional[Path] = None) -> list[dict]:
    with connect(path) as c:
        rows = c.execute(
            "SELECT role, content, created_at FROM messages "
            "WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------- state snapshots ----------

def latest_state(path: Optional[Path] = None) -> Optional[dict]:
    """Return the most recent state snapshot as a dict, or None if none exist."""
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


def clear_facts(path: Optional[Path] = None) -> None:
    with connect(path) as c:
        c.execute("DELETE FROM facts")
