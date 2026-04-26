"""Lemon-side internal-state lifecycle (dyadic-state stage 3).

Lemon now uses the same three-layer schema as the user: traits + characteristic
adaptations + PAD core affect. The asymmetry is in the *dynamics*, not the
shape:

    traits        — hardcoded from `persona.LEMON_TRAITS`. Never inferred.
    adaptations   — baseline from `persona.LEMON_ADAPTATIONS`. Concerns may
                    accumulate during a session, then get re-pegged at
                    session start.
    state         — PAD core affect, nudged each turn by the LLM during
                    pre-reply (dyadic-state stage 2: state-first, response-
                    second). Re-pegs to a session-start baseline so lemon
                    doesn't carry yesterday's drained energy into a new
                    conversation.

Phasic emotion (lemon-side) is NOT modelled at this stage — there's still no
explicit lemon-side phasic event. Adding one is future work.

Mirrors `storage/user_state.py` shape exactly. The only meaningful differences
are session-start re-pegging (lemon does it; user doesn't), the validator
trait clamp (still ±1, but lemon's traits arrive from persona constants and
should never drift), and the load/save pair using a separate db table.
"""
from __future__ import annotations

import copy
from typing import Optional

from logging_setup import get_logger
from persona import LEMON_ADAPTATIONS, LEMON_TRAITS
from storage.db import latest_lemon_state, save_lemon_state_snapshot, latest_state
from storage.user_state import (
    MOOD_LABELS,
    apply_delta,
    validate_delta,
    validate_user_state,
)

log = get_logger("storage.lemon_state")

__all__ = [
    "DEFAULT_LEMON_STATE",
    "LEMON_SESSION_START_STATE",
    "MOOD_LABELS",
    "apply_delta",
    "fresh_lemon_session_state",
    "load_lemon_state",
    "migrate_legacy_state",
    "save_lemon_state",
    "validate_delta",
    "validate_lemon_state",
]


def _build_default_lemon_state() -> dict:
    """Persona-derived default. Traits are LEMON_TRAITS verbatim; adaptations
    are seeded from LEMON_ADAPTATIONS; state is neutral (the session-start
    routine will re-peg it on first load)."""
    return {
        "traits": dict(LEMON_TRAITS),
        "adaptations": {
            "current_goals":     list(LEMON_ADAPTATIONS.get("current_goals") or []),
            "values":            list(LEMON_ADAPTATIONS.get("values") or []),
            "concerns":          list(LEMON_ADAPTATIONS.get("concerns") or []),
            "relational_stance": LEMON_ADAPTATIONS.get("relational_stance"),
        },
        "state": {
            "pleasure":   0.0,
            "arousal":    0.0,
            "dominance":  0.0,
            "mood_label": "neutral",
        },
    }


DEFAULT_LEMON_STATE: dict = _build_default_lemon_state()


# Session-start baseline — lemon "picks up the phone" upbeat-and-warm rather
# than inheriting the drained PAD from the previous session's end. Counterpart
# to the legacy SESSION_START_OVERRIDES from `state.py` translated to PAD.
#
# pleasure +0.30  — pleasant but not euphoric
# arousal   0.00  — neither sleepy nor activated; medium energy
# dominance 0.10  — lightly grounded, present-and-comfortable
# mood_label "content" — folksy label that matches that PAD region
LEMON_SESSION_START_STATE: dict = {
    "pleasure":   0.30,
    "arousal":    0.00,
    "dominance":  0.10,
    "mood_label": "content",
}


# ---------- validators ----------

def validate_lemon_state(parsed: dict, fallback: Optional[dict] = None) -> dict:
    """Same shape rules as the user side; just sources the default differently."""
    return validate_user_state(parsed, fallback=fallback or DEFAULT_LEMON_STATE)


# ---------- legacy migration ----------

# Map old categorical mood values onto a coarse PAD coordinate + new mood_label.
# Used once at startup if there's no lemon_state_snapshot but there IS an old
# state_snapshot, so existing installations don't lose their tonic baseline.
_LEGACY_MOOD_TO_PAD: dict[str, tuple[float, float, float, str]] = {
    "neutral":  (0.00,  0.00,  0.00, "neutral"),
    "good":     (0.30,  0.00,  0.10, "content"),
    "happy":    (0.55,  0.20,  0.15, "happy"),
    "content":  (0.30, -0.10,  0.05, "content"),
    "low":      (-0.30, -0.10, -0.10, "low"),
    "tired":    (-0.20, -0.40, -0.10, "tired"),
    "anxious":  (-0.10,  0.40, -0.30, "anxious"),
    "restless": (-0.05,  0.40, -0.10, "tense"),
}

_LEGACY_ENERGY_TO_AROUSAL: dict[str, float] = {
    "low":    -0.30,
    "medium":  0.00,
    "high":    0.30,
}


def migrate_legacy_state(legacy: dict) -> dict:
    """Convert the old 6-field internal_state shape into the new three-layer shape.

    Used once on startup when there's no lemon_state_snapshot but there is an
    old state_snapshot. Pure function; tests can drive it directly.
    """
    new_state = copy.deepcopy(DEFAULT_LEMON_STATE)

    # PAD + mood_label from legacy mood, blended with arousal from legacy energy
    mood = (legacy or {}).get("mood") or "neutral"
    energy = (legacy or {}).get("energy") or "medium"
    pleasure, arousal, dominance, mood_label = _LEGACY_MOOD_TO_PAD.get(
        mood, _LEGACY_MOOD_TO_PAD["neutral"]
    )
    arousal = max(-1.0, min(1.0, arousal + _LEGACY_ENERGY_TO_AROUSAL.get(energy, 0.0)))
    new_state["state"] = {
        "pleasure":   pleasure,
        "arousal":    arousal,
        "dominance":  dominance,
        "mood_label": mood_label,
    }

    # disposition ("warm" / "normal" / "slightly reserved") -> relational_stance
    disposition = (legacy or {}).get("disposition")
    if disposition:
        new_state["adaptations"]["relational_stance"] = {
            "warm": "warm and present",
            "normal": "present, friendly",
            "slightly reserved": "polite but a little reserved",
        }.get(disposition, "warm and present")

    # emotional_thread → first concern (preserves the "what's quietly on the bot's mind" semantics)
    emotional_thread = (legacy or {}).get("emotional_thread")
    if emotional_thread:
        new_state["adaptations"]["concerns"] = [str(emotional_thread)[:80]]

    # recent_activity is dropped on migration — it was rarely populated and
    # didn't survive the abstraction. Preserved in the legacy rows for archeology.

    return validate_lemon_state(new_state)


# ---------- persistence ----------

def load_lemon_state() -> dict:
    """Load the latest persisted lemon_state. If none exists, look for a legacy
    `state_snapshots` row to migrate. Failing that, return the persona-derived
    default.
    """
    saved = latest_lemon_state()
    if saved is not None:
        return validate_lemon_state(saved, fallback=DEFAULT_LEMON_STATE)

    legacy = latest_state()
    if legacy is not None:
        log.info("legacy_state_migrated mood=%s energy=%s", legacy.get("mood"), legacy.get("energy"))
        migrated = migrate_legacy_state(legacy)
        # Persist immediately so the next load reads from the new table.
        save_lemon_state(migrated)
        return migrated

    return copy.deepcopy(DEFAULT_LEMON_STATE)


def fresh_lemon_session_state() -> dict:
    """Load latest, re-peg the PAD/state layer to LEMON_SESSION_START_STATE,
    and reset relational_stance to the persona baseline. Concerns and goals
    carry over so cross-session continuity (remembering what's quietly on
    lemon's mind) still works."""
    state = load_lemon_state()
    state["state"] = dict(LEMON_SESSION_START_STATE)
    # Reset relational_stance to the persona baseline; concerns/goals persist.
    state["adaptations"]["relational_stance"] = LEMON_ADAPTATIONS.get("relational_stance")
    return state


def save_lemon_state(state: dict, session_id: Optional[int] = None) -> None:
    save_lemon_state_snapshot(state, session_id=session_id)
