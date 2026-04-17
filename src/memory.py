"""Emotional memory: retrieve past user messages tagged with the current emotion.

Used by the empathy pipeline to remind lemon of past moments when the user
felt the same thing. Reduces "we just talked about this" misses across
sessions. Reads from the same `messages` table the chat loop writes to.
"""
from datetime import datetime
from typing import Optional

import db

MEMORY_TAG = "<emotional_memory>"


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


def _humanize_age(created_at: str) -> str:
    """Render an ISO timestamp as 'today', 'yesterday', or 'N days ago'. Best-effort."""
    try:
        ts = datetime.fromisoformat(created_at)
    except (TypeError, ValueError):
        return "earlier"
    days = (datetime.now() - ts).days
    if days <= 0:
        return "earlier today"
    if days == 1:
        return "yesterday"
    if days < 14:
        return f"{days} days ago"
    if days < 60:
        weeks = days // 7
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    months = days // 30
    return f"{months} month{'s' if months != 1 else ''} ago"


def format_memory_block(memories: list[dict]) -> str:
    """Format retrieved memories as an `<emotional_memory>` system block. Empty if no memories."""
    if not memories:
        return ""

    lines = []
    for m in memories:
        when = _humanize_age(m.get("created_at", ""))
        emo = m.get("emotion") or "similar"
        # truncate long messages — we only need the gist
        snippet = m["content"]
        if len(snippet) > 160:
            snippet = snippet[:157].rstrip() + "..."
        lines.append(f"- {when}, when feeling {emo}: \"{snippet}\"")

    body = "\n".join(lines)
    return f"""
<emotional_memory>
Past moments when they felt similar things. Quietly informs your current read of them. Do not bring these up unless it's the natural thing to do.

{body}
</emotional_memory>
""".strip()
