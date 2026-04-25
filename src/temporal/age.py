"""Past-event age humanization.

Renders an ISO timestamp as `today` / `yesterday` / `N days ago` /
`N weeks ago` / `N months ago`. Used by the `<emotional_memory>` block
to date past user messages without showing exact timestamps.

The block formatter that consumes this lives in `prompts.py`.
"""
from datetime import datetime


def humanize_age(created_at: str) -> str:
    """Render an ISO timestamp as a fuzzy human-readable age. Best-effort."""
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
