"""Tiny shared helpers for prompt construction and LLM response parsing.

Both `user_read.py` and `post_exchange.py` build prompts with a "recent
conversation" block and parse responses that sometimes arrive wrapped in
Markdown code fences. The schema-specific validators in `emotion.py`,
`tom.py`, `fact_extractor.py`, and `state.py` also had to strip fences.
This module centralizes both so we have one implementation to maintain.
"""
from __future__ import annotations

from typing import Optional


def strip_json_fences(raw: str) -> str:
    """Remove a single leading ```…``` fence (with optional `json` tag) and
    surrounding whitespace. Safe to call on already-clean JSON strings."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def format_recent_for_prompt(
    recent_msgs: Optional[list[dict]],
    n: int = 6,
    empty_marker: str = "(no prior turns)",
) -> str:
    """Render the last `n` turns as a human-readable block for a classifier
    prompt. Uses "Them" / "You (lemon)" role labels."""
    if not recent_msgs:
        return empty_marker
    lines = []
    for m in recent_msgs[-n:]:
        role = "Them" if m["role"] == "user" else "You (lemon)"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)
