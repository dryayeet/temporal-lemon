"""Pre-generation emotion classifier.

One cheap LLM call per turn (default model: STATE_MODEL — Haiku). Reads the
user's latest message + a few prior turns, returns a structured estimate of
their emotional state. The result is injected as a `<user_emotion>` system
block and stored on the message row in the db.
"""
import json
from typing import Optional

import requests

from config import OPENROUTER_HEADERS, OPENROUTER_URL, STATE_MODEL

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


def _build_prompt(user_msg: str, recent_msgs: Optional[list[dict]]) -> str:
    context_lines = []
    if recent_msgs:
        for m in recent_msgs[-4:]:
            role = "Them" if m["role"] == "user" else "You (lemon)"
            context_lines.append(f"{role}: {m['content']}")
    context = "\n".join(context_lines) if context_lines else "(no prior turns)"

    label_csv = ", ".join(EMOTION_LABELS)
    return f"""
You read the user's latest message and infer their emotional state. You are NOT replying to them — only classifying.

Recent conversation:
{context}

Latest user message:
"{user_msg}"

Return a JSON object with these keys:
- "primary": one of [{label_csv}]
- "intensity": float between 0.0 (very mild) and 1.0 (very strong)
- "underlying_need": short string describing what they probably want from the next reply (e.g. "feel heard, not solved", "be distracted", "get a straight answer"), or null if unclear
- "undertones": list of zero to three secondary emotions from the same label set

Be honest. If the message is flat small-talk, "neutral" with low intensity is the right answer. Do not over-pathologize.

Respond with ONLY the JSON object. No explanation, no markdown.
""".strip()


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


def classify_emotion(
    user_msg: str,
    recent_msgs: Optional[list[dict]] = None,
    model: Optional[str] = None,
) -> dict:
    """Return a structured emotion dict for `user_msg`. Falls back to DEFAULT_EMOTION on any failure."""
    prompt = _build_prompt(user_msg, recent_msgs)
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=OPENROUTER_HEADERS,
            json={
                "model": model or STATE_MODEL,
                "temperature": 0.2,
                "max_tokens": 250,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=20,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        return _parse(raw)

    except requests.HTTPError as e:
        body = getattr(e.response, "text", "")[:300]
        print(f"  [emotion classifier http error: {e} | body: {body}]")
        return dict(DEFAULT_EMOTION)
    except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"  [emotion classifier failed: {e}]")
        return dict(DEFAULT_EMOTION)


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
