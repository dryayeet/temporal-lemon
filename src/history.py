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
        "content": (
            "<earlier_conversation>\n"
            "Here is a rough record of what was said earlier in this chat. "
            "It is not recent but it is part of your shared history with this person. "
            "Reference it only if it comes up naturally, not to fill silence.\n\n"
            f"{old_text}\n"
            "</earlier_conversation>"
        ),
    }

    return system_msgs + [summary_block] + recent_msgs
