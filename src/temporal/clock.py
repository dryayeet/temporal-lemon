"""Clock-time labelers used by the `<time_context>` system block.

Bucket the current hour and the elapsed-session-minutes into the small
descriptive strings the prompt embeds. Pure functions, no I/O.

The `<time_context>` formatter that consumes these lives in `prompts.py`.
"""


def time_of_day_label(hour: int) -> str:
    if 5 <= hour < 10:
        return "morning"
    if 10 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    if 21 <= hour < 24:
        return "late night"
    return "very late night / early hours"


def session_duration_note(elapsed_minutes: int) -> str:
    if elapsed_minutes < 2:
        return "This conversation just started."
    if elapsed_minutes < 10:
        return f"You've been talking for about {elapsed_minutes} minutes."
    if elapsed_minutes < 30:
        return f"You've been talking for a bit now, around {elapsed_minutes} minutes."
    return f"This has been a long conversation, going on for about {elapsed_minutes} minutes."
