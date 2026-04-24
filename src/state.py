"""Internal-state lifecycle: defaults, parsing, formatting, persistence, updater.

State persists in SQLite (see db.py). The chat loop loads the latest snapshot
on startup and saves a new snapshot every time the state changes.
"""
import json
from typing import Optional

import requests

from config import OPENROUTER_HEADERS, OPENROUTER_URL, STATE_MODEL
from db import latest_state, save_state_snapshot

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


def format_internal_state(state: dict) -> str:
    thread = state["emotional_thread"] or "nothing specific"
    activity = state["recent_activity"] or "nothing worth mentioning"

    return f"""
<internal_state>
This is your current internal state. It is the reason behind how you text, not something you talk about.
Your responses should naturally reflect this state without ever naming it.

Mood: {state["mood"]}
Energy: {state["energy"]}
Engagement level: {state["engagement"]}
What's on your mind: {thread}
What you've been up to: {activity}
Disposition toward this person right now: {state["disposition"]}

Let this shape your word choice, reply length, warmth, and how much you push the conversation.
Do not perform these states. Just let them bleed through naturally.
</internal_state>
""".strip()


def parse_state_response(raw: str, fallback: dict) -> dict:
    """Parse a JSON state blob from the model, tolerating fenced code blocks."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    updated = json.loads(raw)
    if not isinstance(updated, dict):
        raise ValueError("state response was not a JSON object")
    return {k: updated.get(k, fallback[k]) for k in DEFAULT_STATE}


def update_internal_state(state: dict, user_msg: str, bot_reply: str) -> dict:
    """Ask a lightweight model to nudge the internal state based on the latest exchange."""
    prompt = f"""
You are managing the internal emotional state of a chatbot that simulates a human friend.

Current state:
{json.dumps(state, indent=2)}

What just happened in the conversation:
User said: "{user_msg}"
Bot replied: "{bot_reply}"

Based on this exchange, suggest small, realistic updates to the internal state.
Rules:
- Changes should be subtle nudges, not dramatic shifts.
- Only change fields where the conversation genuinely warrants it.
- mood and energy shift slowly. A single message rarely changes them much.
- engagement should reflect how present and interested the user seems right now.
- emotional_thread should capture anything that seems to be on the bot's mind after this exchange. Can be null.
- recent_activity should only be set if the conversation has causally established something the bot has been doing. Do not invent.
- disposition shifts only if the user's tone or behavior warrants it.

Respond ONLY with a valid JSON object with the same keys as the current state. No explanation. No markdown.
"""

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=OPENROUTER_HEADERS,
            json={
                "model": STATE_MODEL,
                "temperature": 0.3,
                "max_tokens": 400,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        return parse_state_response(raw, fallback=state)

    except requests.HTTPError as e:
        body = getattr(e.response, "text", "")[:400]
        print(f"  [state update http error: {e} | body: {body}]")
        return state
    except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"  [state update failed: {e}]")
        return state
