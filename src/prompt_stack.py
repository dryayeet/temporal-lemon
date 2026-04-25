"""System-block stack helpers.

These functions manipulate the list of messages that make up a chat history,
inserting/replacing the tagged system blocks lemon uses for time/state/facts/
etc. They are not prompts themselves — the prompt strings they reference live
in `prompts.py`. Kept separate so `prompts.py` stays a pure content file.
"""
from __future__ import annotations

from prompts import format_earlier_conversation


def replace_system_block(history: list, tag: str, content: str, position: int) -> list:
    """Drop any existing system message containing `tag`, insert `content` at `position`."""
    filtered = [
        m for m in history
        if not (m["role"] == "system" and tag in m["content"])
    ]
    filtered.insert(position, {"role": "system", "content": content})
    return filtered


def compress_history(history: list, keep_recent: int) -> list:
    """Keep the last `keep_recent` non-system turns verbatim; fold older turns into a summary block."""
    system_msgs = [m for m in history if m["role"] == "system"]
    convo_msgs = [m for m in history if m["role"] != "system"]

    if len(convo_msgs) <= keep_recent:
        return history

    old_msgs = convo_msgs[:-keep_recent]
    recent_msgs = convo_msgs[-keep_recent:]

    old_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in old_msgs
    )

    summary_block = {
        "role": "system",
        "content": format_earlier_conversation(old_text),
    }

    return system_msgs + [summary_block] + recent_msgs
