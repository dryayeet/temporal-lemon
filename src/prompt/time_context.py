from datetime import datetime


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


def get_time_context(session_start: datetime, now: datetime | None = None) -> str:
    now = now or datetime.now()
    elapsed_minutes = int((now - session_start).total_seconds() / 60)
    return f"""
<time_context>
Current local date: {now.strftime('%Y-%m-%d')}
Current local time: {now.strftime('%H:%M')}
Day of week: {now.strftime('%A')}
Time of day: {time_of_day_label(now.hour)}
{session_duration_note(elapsed_minutes)}
</time_context>
""".strip()
