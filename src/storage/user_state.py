"""User-side internal-state lifecycle: defaults, validation, delta application,
persistence.

Stage 1 of the dyadic-state architecture (see `docs/dyadic_state.md`). Mirrors
the lemon-side `state.py` but on the *user* side, with the three-layer schema:

    traits        — Big 5 (OCEAN), each in [-1, +1]. Slow drift, essentially
                    frozen at stage 1.
    adaptations   — current_goals / values / concerns / relational_stance.
                    Medium-term; LLM may add or remove single entries per turn.
    state         — PAD core affect (pleasure / arousal / dominance), each in
                    [-1, +1], plus a derived categorical mood label for prompt
                    readability. Nudged each turn by the phasic event read in
                    `empathy/user_read.py`.

The phasic layer (the 23-label categorical event with intensity) is NOT stored
here — it stays per-message on the `messages` row. This module is for the
tonic state only.

The validator and `apply_delta` follow the project's "clamp / whitelist /
default rather than reject" pattern (see `empathy/emotion._validate`).
"""
from __future__ import annotations

import copy
from typing import Optional

from llm.parse_utils import strip_json_fences  # noqa: F401  (kept for parity / future use)
from logging_setup import get_logger
from storage.db import latest_user_state, save_user_state_snapshot

log = get_logger("storage.user_state")


# ---------- canonical schema ----------

DEFAULT_USER_STATE: dict = {
    "traits": {
        "openness":          0.0,
        "conscientiousness": 0.0,
        "extraversion":      0.0,
        "agreeableness":     0.0,
        "neuroticism":       0.0,
    },
    "adaptations": {
        "current_goals":     [],
        "values":            [],
        "concerns":          [],
        "relational_stance": None,
    },
    "state": {
        "pleasure":   0.0,
        "arousal":    0.0,
        "dominance":  0.0,
        "mood_label": "neutral",
    },
}

# Small categorical set the LLM picks `mood_label` from. Derived view of PAD
# coordinates; chosen folksy enough that it reads well in the system prompt.
MOOD_LABELS: list[str] = [
    "neutral",
    "calm",
    "content",
    "happy",
    "excited",
    "anxious",
    "low",
    "tense",
    "tired",
    "frustrated",
]

_TRAIT_KEYS = tuple(DEFAULT_USER_STATE["traits"].keys())
_PAD_KEYS = ("pleasure", "arousal", "dominance")

# Per-turn caps so the user-state evolves smoothly even when the LLM is bold.
# These are the "subtle nudges only" enforcement at the validator layer.
_PAD_NUDGE_CAP = 0.15      # how far PAD can move per turn
_TRAIT_NUDGE_CAP = 0.02    # traits are essentially frozen at stage 1
_LIST_MAX_LEN = 5          # cap on goals/values/concerns
_STRING_MAX_LEN = 80       # per-entry length cap for adaptations


# ---------- helpers ----------

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _coerce_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_string(value, max_len: int = _STRING_MAX_LEN) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    s = value.strip()
    if not s:
        return None
    return s[:max_len].rstrip()


def _clean_string_list(values, max_items: int = _LIST_MAX_LEN) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for v in values:
        s = _clean_string(v)
        if s is None:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s)
        if len(cleaned) >= max_items:
            break
    return cleaned


# ---------- validators ----------

def validate_user_state(parsed: dict, fallback: Optional[dict] = None) -> dict:
    """Coerce a parsed dict into the canonical user-state schema.

    Missing fields fall back to `fallback` (defaults to DEFAULT_USER_STATE).
    Out-of-range values are clamped; unknown values are dropped or replaced
    with defaults. Never raises on shape — only on `parsed` not being a dict.
    """
    if not isinstance(parsed, dict):
        raise ValueError("user_state response was not a JSON object")

    fb = copy.deepcopy(fallback if fallback is not None else DEFAULT_USER_STATE)

    traits_in = parsed.get("traits") if isinstance(parsed.get("traits"), dict) else {}
    traits = {
        k: _clamp(_coerce_float(traits_in.get(k, fb["traits"].get(k, 0.0))), -1.0, 1.0)
        for k in _TRAIT_KEYS
    }

    adapt_in = parsed.get("adaptations") if isinstance(parsed.get("adaptations"), dict) else {}
    adaptations = {
        "current_goals":     _clean_string_list(adapt_in.get("current_goals", fb["adaptations"]["current_goals"])),
        "values":            _clean_string_list(adapt_in.get("values", fb["adaptations"]["values"])),
        "concerns":          _clean_string_list(adapt_in.get("concerns", fb["adaptations"]["concerns"])),
        "relational_stance": _clean_string(adapt_in.get("relational_stance", fb["adaptations"]["relational_stance"])),
    }

    state_in = parsed.get("state") if isinstance(parsed.get("state"), dict) else {}
    pad = {
        k: _clamp(_coerce_float(state_in.get(k, fb["state"].get(k, 0.0))), -1.0, 1.0)
        for k in _PAD_KEYS
    }
    mood_label = state_in.get("mood_label", fb["state"].get("mood_label", "neutral"))
    if not isinstance(mood_label, str) or mood_label not in MOOD_LABELS:
        mood_label = "neutral"

    return {
        "traits":      traits,
        "adaptations": adaptations,
        "state":       {**pad, "mood_label": mood_label},
    }


def validate_delta(parsed) -> dict:
    """Coerce a parsed delta dict into the canonical delta schema.

    Delta semantics:
        pad             — small per-axis nudges (clamped to ±_PAD_NUDGE_CAP)
        mood_label      — whitelisted picked label (None means no change)
        trait_nudges    — per-trait nudge (clamped to ±_TRAIT_NUDGE_CAP)
        goal_add        — short strings to add to current_goals
        goal_remove     — short strings to remove from current_goals
        concern_add     — short strings to add to concerns
        concern_remove  — short strings to remove from concerns
        value_add       — short strings to add to values
        stance          — replacement relational_stance (None means no change)

    Anything missing or wrong-typed becomes a no-op. Returning an
    all-zero/empty delta on any failure is the safe fallback.
    """
    zero: dict = {
        "pad":            {k: 0.0 for k in _PAD_KEYS},
        "mood_label":     None,
        "trait_nudges":   {},
        "goal_add":       [],
        "goal_remove":    [],
        "concern_add":    [],
        "concern_remove": [],
        "value_add":      [],
        "stance":         None,
    }

    if not isinstance(parsed, dict):
        return zero

    pad_in = parsed.get("pad") if isinstance(parsed.get("pad"), dict) else {}
    pad = {
        k: _clamp(_coerce_float(pad_in.get(k, 0.0)), -_PAD_NUDGE_CAP, _PAD_NUDGE_CAP)
        for k in _PAD_KEYS
    }

    mood_label = parsed.get("mood_label")
    if not isinstance(mood_label, str) or mood_label not in MOOD_LABELS:
        mood_label = None

    trait_in = parsed.get("trait_nudges") if isinstance(parsed.get("trait_nudges"), dict) else {}
    trait_nudges = {
        k: _clamp(_coerce_float(v, 0.0), -_TRAIT_NUDGE_CAP, _TRAIT_NUDGE_CAP)
        for k, v in trait_in.items()
        if k in _TRAIT_KEYS
    }

    return {
        "pad":            pad,
        "mood_label":     mood_label,
        "trait_nudges":   trait_nudges,
        "goal_add":       _clean_string_list(parsed.get("goal_add"), max_items=2),
        "goal_remove":    _clean_string_list(parsed.get("goal_remove"), max_items=2),
        "concern_add":    _clean_string_list(parsed.get("concern_add"), max_items=2),
        "concern_remove": _clean_string_list(parsed.get("concern_remove"), max_items=2),
        "value_add":      _clean_string_list(parsed.get("value_add"), max_items=1),
        "stance":         _clean_string(parsed.get("stance")),
    }


# ---------- delta application ----------

def apply_delta(prev: dict, delta: dict) -> dict:
    """Return a new user_state dict with `delta` applied to `prev`.

    Pure function: does not mutate `prev`. Always returns a fully validated
    state shape (so even a buggy delta can't poison the snapshot).
    """
    new_state = copy.deepcopy(prev) if prev else copy.deepcopy(DEFAULT_USER_STATE)

    # PAD nudges
    for k in _PAD_KEYS:
        nudge = float(delta.get("pad", {}).get(k, 0.0))
        new_state["state"][k] = _clamp(new_state["state"][k] + nudge, -1.0, 1.0)

    # mood label override (only when explicitly set)
    if delta.get("mood_label") in MOOD_LABELS:
        new_state["state"]["mood_label"] = delta["mood_label"]

    # trait nudges (essentially frozen — caller can re-clamp if needed)
    for k, nudge in (delta.get("trait_nudges") or {}).items():
        if k not in _TRAIT_KEYS:
            continue
        new_state["traits"][k] = _clamp(new_state["traits"][k] + float(nudge), -1.0, 1.0)

    # adaptation list updates
    def _apply_list(field: str, adds: list[str], removes: list[str]) -> list[str]:
        existing = list(new_state["adaptations"].get(field) or [])
        # remove first so caller can replace within one turn
        if removes:
            lower_removes = {s.lower() for s in removes}
            existing = [s for s in existing if s.lower() not in lower_removes]
        # then add (deduped, capped)
        for s in adds:
            if s.lower() in {x.lower() for x in existing}:
                continue
            existing.append(s)
            if len(existing) >= _LIST_MAX_LEN:
                break
        return existing

    new_state["adaptations"]["current_goals"] = _apply_list(
        "current_goals",
        delta.get("goal_add") or [],
        delta.get("goal_remove") or [],
    )
    new_state["adaptations"]["concerns"] = _apply_list(
        "concerns",
        delta.get("concern_add") or [],
        delta.get("concern_remove") or [],
    )
    new_state["adaptations"]["values"] = _apply_list(
        "values",
        delta.get("value_add") or [],
        [],
    )

    if delta.get("stance"):
        new_state["adaptations"]["relational_stance"] = delta["stance"]

    return validate_user_state(new_state)


def is_cold_start(state: dict) -> bool:
    """True when `state` is the all-default shape — no inferred signal yet."""
    if not state:
        return True
    s = state.get("state") or {}
    if abs(s.get("pleasure", 0.0)) > 1e-6:
        return False
    if abs(s.get("arousal", 0.0)) > 1e-6:
        return False
    if abs(s.get("dominance", 0.0)) > 1e-6:
        return False
    if s.get("mood_label", "neutral") != "neutral":
        return False
    adapt = state.get("adaptations") or {}
    if any(adapt.get(k) for k in ("current_goals", "values", "concerns")):
        return False
    if adapt.get("relational_stance"):
        return False
    return True


# ---------- persistence ----------

def load_user_state() -> dict:
    """Load the latest persisted user_state, falling back to DEFAULT_USER_STATE."""
    saved = latest_user_state()
    if saved is None:
        return copy.deepcopy(DEFAULT_USER_STATE)
    return validate_user_state(saved, fallback=DEFAULT_USER_STATE)


def fresh_user_session_state() -> dict:
    """The user side has NO session-start overrides — they bring whatever they
    carried in. So this is just `load_user_state()`. (Symmetric structure with
    `state.fresh_session_state` but asymmetric dynamics by design — see
    `docs/dyadic_state.md` §7.4.)
    """
    return load_user_state()


def save_user_state(state: dict, session_id: Optional[int] = None) -> None:
    save_user_state_snapshot(state, session_id=session_id)
