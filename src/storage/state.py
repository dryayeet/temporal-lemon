"""DEPRECATED — kept only for tests in `tests/test_state.py`.

The 6-field internal_state schema (mood / energy / engagement / emotional_thread
/ recent_activity / disposition) was retired in stage 3 of the dyadic-state
architecture. Lemon's runtime tonic state now uses the same three-layer schema
as the user side: traits + adaptations + PAD core affect. See
`storage/lemon_state.py` for the live module and `docs/dyadic_state.md` §6 for
the new schema and rationale.

The legacy `state_snapshots` SQLite table stays in the schema as archive but
gets no new writes. This module's `validate_state` / `parse_state_response`
helpers are still exercised by the legacy migration path in
`storage.lemon_state.migrate_legacy_state` (indirectly) and by
`tests/test_state.py`, so they are preserved here intentionally. New code
should not import from this module.
"""
import json
from typing import Optional

from llm.parse_utils import strip_json_fences
from storage.db import latest_state, save_state_snapshot

DEFAULT_STATE = {
    "mood": "neutral",          # neutral | good | low | happy | anxious | restless | tired | content
    "energy": "medium",         # low | medium | high
    "engagement": "normal",     # low | normal | deep
    "emotional_thread": None,   # string or null
    "recent_activity": None,    # string or null
    "disposition": "warm",      # warm | normal | slightly reserved
}

SESSION_START_OVERRIDES = {
    "mood": "good",
    "energy": "medium",
    "engagement": "normal",
    "disposition": "warm",
}


def load_state() -> dict:
    saved = latest_state()
    if saved is None:
        return dict(DEFAULT_STATE)
    merged = dict(DEFAULT_STATE)
    merged.update({k: v for k, v in saved.items() if k in DEFAULT_STATE})
    return merged


def fresh_session_state() -> dict:
    state = load_state()
    state.update(SESSION_START_OVERRIDES)
    return state


def save_state(state: dict, session_id: Optional[int] = None) -> None:
    save_state_snapshot(state, session_id=session_id)


def validate_state(parsed: dict, fallback: dict) -> dict:
    if not isinstance(parsed, dict):
        raise ValueError("state response was not a JSON object")
    return {k: parsed.get(k, fallback[k]) for k in DEFAULT_STATE}


def parse_state_response(raw: str, fallback: dict) -> dict:
    return validate_state(json.loads(strip_json_fences(raw)), fallback)
