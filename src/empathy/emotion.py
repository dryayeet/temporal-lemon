"""Emotion schema + validator + family clustering.

The label whitelist, default shape, validator, and JSON parser. The
LLM call lives in `user_read.py` (merged with theory-of-mind into a
single pre-generation round-trip). The `<user_emotion>` system-block
formatter and the `EMOTION_LABELS` whitelist live in `prompts.py`.

`EMOTION_FAMILIES` and `emotion_relatedness` cluster related labels
together (sad/lonely/grief, anger/frustration, etc.) so the memory
retriever can give a partial mood-congruence boost when the current
label and a past label share a family without being identical.
Grouping follows the valence/family conventions used in the empathy
literature (Plutchik wheel, GoEmotions cluster maps).
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


# Family map keyed by label. Every label in EMOTION_LABELS must appear here;
# `_check_families_complete()` enforces that at import time.
EMOTION_FAMILIES: dict[str, str] = {
    # neutral
    "neutral":         "neutral",
    # positive valence
    "joy":             "positive",
    "excitement":      "positive",
    "love":            "positive",
    "gratitude":       "positive",
    "amused":          "positive",
    # sadness cluster
    "sadness":         "sad",
    "loneliness":      "sad",
    "disappointment":  "sad",
    "grief":           "sad",
    # anger cluster
    "anger":           "anger",
    "frustration":     "anger",
    "annoyance":       "anger",
    # fear / unease cluster
    "fear":            "fear",
    "anxiety":         "fear",
    "confusion":       "fear",
    # self-conscious cluster
    "shame":           "shame",
    "embarrassment":   "shame",
    "guilt":           "shame",
    # arousal-only
    "tired":           "low_arousal",
    "curious":         "exploratory",
}


def _check_families_complete() -> None:
    """Surface schema drift early — every label must have a family."""
    missing = set(EMOTION_LABELS) - set(EMOTION_FAMILIES)
    if missing:
        raise RuntimeError(f"EMOTION_FAMILIES missing labels: {sorted(missing)}")


_check_families_complete()


def family_of(label: str) -> str:
    """Return the family bucket for an emotion label, or `unknown`."""
    return EMOTION_FAMILIES.get(label, "unknown")


def emotion_relatedness(a: str, b: str) -> float:
    """How emotionally related two labels are, on [0, 1].

    1.0  — same exact label
    0.5  — different label, same family (e.g. sadness vs loneliness)
    0.0  — different family, or either label is `neutral` / `unknown`

    Used by the memory retriever to give a partial mood-congruence boost.
    Neutral never confers a bonus (every neutral message looking at every
    other neutral message would dominate the score).
    """
    if not a or not b:
        return 0.0
    if a == b:
        # Neutral exact-match doesn't earn a bonus either — too cheap.
        return 0.0 if a == "neutral" else 1.0
    fam_a = family_of(a)
    fam_b = family_of(b)
    if fam_a == "unknown" or fam_b == "unknown":
        return 0.0
    if fam_a == "neutral" or fam_b == "neutral":
        return 0.0
    return 0.5 if fam_a == fam_b else 0.0


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
