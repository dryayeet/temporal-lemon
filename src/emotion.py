"""Emotion schema + parser + system-block formatter.

The LLM call itself now lives in `user_read.py` (merged with theory-of-mind
into a single pre-generation round-trip). This module keeps the label
whitelist, the validator (`_parse`), the default shape, and the
`<user_emotion>` system-block formatter — all of which user_read and the
pipeline still consume.
"""
import json

EMOTION_TAG = "<user_emotion>"

# Label set inspired by GoEmotions but trimmed to what's discriminable in
# friend-chat context. The classifier is asked to pick one.
EMOTION_LABELS = [
    "neutral", "joy", "excitement", "love", "gratitude",
    "sadness", "loneliness", "disappointment", "grief",
    "anger", "frustration", "annoyance",
    "fear", "anxiety", "confusion",
    "shame", "embarrassment", "guilt",
    "tired", "amused", "curious",
]

DEFAULT_EMOTION = {
    "primary": "neutral",
    "intensity": 0.3,
    "underlying_need": None,
    "undertones": [],
}


def _parse(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("emotion response was not a JSON object")

    primary = str(parsed.get("primary", "neutral")).strip().lower()
    if primary not in EMOTION_LABELS:
        primary = "neutral"

    try:
        intensity = float(parsed.get("intensity", 0.0))
    except (TypeError, ValueError):
        intensity = 0.0
    intensity = max(0.0, min(1.0, intensity))

    need = parsed.get("underlying_need")
    if need is not None and not isinstance(need, str):
        need = None
    if isinstance(need, str):
        need = need.strip() or None

    undertones_raw = parsed.get("undertones") or []
    if not isinstance(undertones_raw, list):
        undertones_raw = []
    undertones = [
        str(u).strip().lower() for u in undertones_raw[:3]
        if isinstance(u, (str, int)) and str(u).strip().lower() in EMOTION_LABELS
    ]

    return {
        "primary": primary,
        "intensity": intensity,
        "underlying_need": need,
        "undertones": undertones,
    }


def format_emotion_block(emotion: dict) -> str:
    """Format an emotion dict as a `<user_emotion>` system block for injection."""
    primary = emotion.get("primary", "neutral")
    intensity = emotion.get("intensity", 0.0)
    need = emotion.get("underlying_need")
    undertones = emotion.get("undertones") or []

    need_line = f"What they probably want: {need}" if need else "What they probably want: unclear"
    undertone_line = (
        f"Undertones: {', '.join(undertones)}" if undertones else "Undertones: none"
    )
    intensity_word = (
        "mild" if intensity < 0.3
        else "moderate" if intensity < 0.6
        else "strong" if intensity < 0.85
        else "very strong"
    )

    return f"""
<user_emotion>
A separate read of the user's last message before you reply. Treat this as background — do not name or quote it.

Primary feeling: {primary} ({intensity_word}, intensity {intensity:.2f})
{undertone_line}
{need_line}

Let this shape your tone, length, and whether to ask vs. acknowledge. Do not echo the label back.
</user_emotion>
""".strip()
