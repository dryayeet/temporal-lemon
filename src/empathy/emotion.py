"""Emotion schema + validator.

The label whitelist, default shape, validator, and JSON parser. The
LLM call lives in `user_read.py` (merged with theory-of-mind into a
single pre-generation round-trip). The `<user_emotion>` system-block
formatter and the `EMOTION_LABELS` whitelist live in `prompts.py`.
"""
import json

from llm.parse_utils import strip_json_fences
from prompts import EMOTION_LABELS

DEFAULT_EMOTION = {
    "primary": "neutral",
    "intensity": 0.3,
    "underlying_need": None,
    "undertones": [],
}


def _validate(parsed: dict) -> dict:
    """Coerce an already-parsed emotion dict into the canonical schema.

    LLMs occasionally produce out-of-range intensities, unknown labels, or
    wrong types — we clamp/whitelist/default rather than reject.
    """
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


def _parse(raw: str) -> dict:
    """Parse a raw LLM response string (optionally fenced) into the canonical dict."""
    return _validate(json.loads(strip_json_fences(raw)))
