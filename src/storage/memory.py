"""Emotional memory: retrieve past user messages tagged with the current emotion.

Used by the empathy pipeline to remind lemon of past moments when the user
felt the same thing. Reduces "we just talked about this" misses across
sessions. Reads from the same `messages` table the chat loop writes to.

The `<emotional_memory>` system-block formatter lives in `prompts.py`.
"""
from typing import Optional

from storage import db


def relevant_memories(
    emotion: str,
    current_session_id: Optional[int] = None,
    limit: int = 3,
) -> list[dict]:
    """Past user messages tagged with `emotion`, excluding the current session, newest first."""
    if not emotion or emotion == "neutral":
        return []
    return db.find_messages_by_emotion(
        emotion=emotion,
        exclude_session_id=current_session_id,
        limit=limit,
    )
