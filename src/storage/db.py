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
from logging_setup import get_logger

log = get_logger("storage.db")

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

-- Full-text-search index over messages.content. External-content table so
-- the source of truth stays in `messages` and we don't pay storage twice.
-- Porter tokenizer enables stemming (talk/talking/talked all match).
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content='messages',
    content_rowid='id',
    tokenize='porter unicode61'
);

-- Keep messages_fts in sync with messages. Standard external-content recipe.
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
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


def _rebuild_fts_if_needed(conn: sqlite3.Connection) -> None:
    """Backfill the FTS5 index for pre-existing message rows.

    External-content FTS5 tables don't auto-populate from existing rows
    (the triggers only catch new inserts), so any DB that predates the
    FTS schema needs a one-shot `rebuild`. Cheap idempotent check: if
    FTS rowcount lags behind messages, rebuild.
    """
    msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    if msg_count == 0:
        return
    fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
    if fts_count >= msg_count:
        return
    conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
    log.info("db_fts_rebuilt msgs=%d", msg_count)


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
    _rebuild_fts_if_needed(conn)
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
        sid = cur.lastrowid
        log.info("session_start id=%s", sid)
        return sid


def end_session(session_id: int, path: Optional[Path] = None) -> None:
    with connect(path) as c:
        c.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            (_now(), session_id),
        )
        log.info("session_end id=%s", session_id)


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
        mid = cur.lastrowid
        log.info(
            "msg_insert id=%s role=%s chars=%d emotion=%s",
            mid, role, len(content), emotion,
        )
        return mid


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
        log.debug(
            "msg_by_emotion emotion=%s returned=%d", emotion, len(rows),
        )
        return [dict(r) for r in rows]


def find_messages_by_fts(
    fts_query: str,
    exclude_session_id: Optional[int] = None,
    candidate_pool: int = 50,
    path: Optional[Path] = None,
) -> list[dict]:
    """Return user messages matching `fts_query` via FTS5, with BM25 scores.

    `fts_query` follows the FTS5 syntax — typically a space- or OR-joined
    list of stemmed tokens (`exam OR tuesday OR prep`). The composite
    scorer in `storage/memory.py` consumes this candidate pool and
    re-ranks with recency + intensity + emotion-relatedness.

    BM25 in FTS5 is signed so that smaller-is-better; we expose the raw
    value and let the scorer normalize. Limit is the *candidate pool* —
    aim large here (50 is plenty) so the composite scorer has options.
    """
    sql = (
        "SELECT m.id, m.session_id, m.role, m.content, m.created_at, "
        "       m.emotion, m.intensity, m.salience, "
        "       bm25(messages_fts) AS bm25 "
        "FROM messages m "
        "JOIN messages_fts ON messages_fts.rowid = m.id "
        "WHERE messages_fts MATCH ? "
        "  AND m.role = 'user' "
    )
    params: list = [fts_query]
    if exclude_session_id is not None:
        sql += "AND m.session_id != ? "
        params.append(exclude_session_id)
    sql += "ORDER BY bm25 LIMIT ?"
    params.append(candidate_pool)

    with connect(path) as c:
        try:
            rows = c.execute(sql, params).fetchall()
        except sqlite3.OperationalError as e:
            # Malformed FTS query (e.g. all stopwords got stripped to "").
            log.warning("fts_invalid query=%r error=%r", fts_query, e)
            return []
    log.debug("fts query=%r returned=%d", fts_query, len(rows))
    return [dict(r) for r in rows]


def find_recent_user_messages(
    exclude_session_id: Optional[int] = None,
    limit: int = 50,
    path: Optional[Path] = None,
) -> list[dict]:
    """Recent user messages (newest first), used as fallback when FTS yields nothing.

    Returned shape matches `find_messages_by_fts` minus the bm25 column,
    so the composite scorer can still rank by recency / intensity /
    emotion-relatedness.
    """
    sql = (
        "SELECT id, session_id, role, content, created_at, emotion, intensity, salience "
        "FROM messages WHERE role = 'user' "
    )
    params: list = []
    if exclude_session_id is not None:
        sql += "AND session_id != ? "
        params.append(exclude_session_id)
    sql += "ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with connect(path) as c:
        rows = c.execute(sql, params).fetchall()
    log.debug("recent_users returned=%d", len(rows))
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
        log.info(
            "state_save mood=%s disposition=%s",
            state.get("mood"), state.get("disposition"),
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
        log.info("fact_upsert key=%s", key)


def get_facts(path: Optional[Path] = None) -> dict[str, str]:
    with connect(path) as c:
        rows = c.execute("SELECT key, value FROM facts ORDER BY key").fetchall()
        return {r["key"]: r["value"] for r in rows}


def delete_fact(key: str, path: Optional[Path] = None) -> bool:
    with connect(path) as c:
        cur = c.execute("DELETE FROM facts WHERE key = ?", (key,))
        deleted = cur.rowcount > 0
        log.info("fact_delete key=%s deleted=%s", key, deleted)
        return deleted
