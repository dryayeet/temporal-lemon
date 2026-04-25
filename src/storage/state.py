"""Internal-state lifecycle: defaults, parsing, persistence.

State persists in SQLite (see `storage/db.py`). The chat loop loads the latest
snapshot on startup and saves a new snapshot every time the state changes.
The LLM call that nudges the state lives in `empathy/post_exchange.py` (merged
with fact extraction into a single post-generation round-trip). The
`<internal_state>` system-block formatter lives in `prompts.py`.
"""
import json
from typing import Optional

from llm.parse_utils import strip_json_fences
from storage.db import latest_state, save_state_snapshot

DEFAULT_STATE = {
    "mood": "neutral",          # neutral | good | low | happy | anxious | restless | tired | content
    "energy": "medium",         # low | medium | high
    "engagement": "normal",     # low | normal | deep
    "emotional_thread": None,   # string or null — quietly on the bot's mind
    "recent_activity": None,    # string or null — only when conversation grounded it
    "disposition": "warm",      # warm | normal | slightly reserved
}

# Overrides applied when a NEW session starts. Simulates "picking up the phone
# to chat" — lemon shouldn't inherit last session's drained energy. The
# emotional_thread and recent_activity fields are preserved so cross-session
# continuity (remembering what was on lemon's mind) still works.
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
    """Load the latest snapshot, then re-peg energy/engagement/mood/disposition
    to an upbeat baseline so every new session starts engaged.
    Cross-session fields (emotional_thread, recent_activity) carry over."""
    state = load_state()
    state.update(SESSION_START_OVERRIDES)
    return state


def save_state(state: dict, session_id: Optional[int] = None) -> None:
    save_state_snapshot(state, session_id=session_id)


def validate_state(parsed: dict, fallback: dict) -> dict:
    """Coerce an already-parsed state dict into the canonical schema.

    Missing keys fall back to the corresponding `fallback` value; extra keys
    are dropped.
    """
    if not isinstance(parsed, dict):
        raise ValueError("state response was not a JSON object")
    return {k: parsed.get(k, fallback[k]) for k in DEFAULT_STATE}


def parse_state_response(raw: str, fallback: dict) -> dict:
    """Parse a raw LLM response string (optionally fenced) into a state dict."""
    return validate_state(json.loads(strip_json_fences(raw)), fallback)
