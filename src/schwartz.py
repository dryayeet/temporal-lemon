"""Schwartz's 10 universal values (Schwartz 1992).

Used to tag entries in the `values` slot of the three-layer state schema.
Each entry in `state.adaptations.values` is a dict `{"label": <free-form
short string>, "schwartz": <one of SCHWARTZ_VALUES or None>}`.

The ten values are organized in a circumplex where adjacent values are
compatible and opposite values conflict. Higher-order axes:

  Self-Transcendence (universalism, benevolence)
        vs.
  Self-Enhancement   (achievement, power, hedonism)

  Openness to Change (self_direction, stimulation)
        vs.
  Conservation       (tradition, conformity, security)

Reference: Schwartz, *Universals in the content and structure of values:
Theoretical advances and empirical tests in 20 countries*, Advances in
Experimental Social Psychology vol. 25, 1992. See also `docs/dyadic_state.md`.
"""
from __future__ import annotations

from typing import Optional


# Tuple ordering follows the circumplex from openness-to-change clockwise
# through self-enhancement, conservation, and self-transcendence.
SCHWARTZ_VALUES: tuple[str, ...] = (
    "self_direction",
    "stimulation",
    "hedonism",
    "achievement",
    "power",
    "security",
    "conformity",
    "tradition",
    "benevolence",
    "universalism",
)


# Short, prompt-ready descriptions for each category. Used inline in
# `prompts.build_user_read_prompt` so the LLM has a quick reference.
SCHWARTZ_DESCRIPTIONS: dict[str, str] = {
    "self_direction": "independent thought and action; creativity, freedom, choosing one's own way",
    "stimulation":    "excitement, novelty, challenge in life",
    "hedonism":       "pleasure, sensuous gratification, enjoying life",
    "achievement":    "personal success through demonstrating competence",
    "power":          "social status, prestige, control over people or resources",
    "security":       "safety, harmony, stability of self / society / relationships",
    "conformity":     "restraint of actions that might upset others or violate norms",
    "tradition":      "respect for customs, family heritage, cultural or religious continuity",
    "benevolence":    "preserving and enhancing welfare of close others (family, friends, partners)",
    "universalism":   "understanding, appreciation, tolerance, fairness for all people and nature",
}


# Aliases the LLM might use that we want to coerce to canonical tags.
# Lowercased; hyphens and spaces are normalized to underscores in the lookup.
_ALIASES: dict[str, str] = {
    "self direction":     "self_direction",
    "self-direction":     "self_direction",
    "selfdirection":      "self_direction",
    "self_transcendence": "universalism",   # not a value per se; coerce to nearest
    "self transcendence": "universalism",
    "self_enhancement":   "achievement",
    "self enhancement":   "achievement",
    "openness":           "self_direction",  # closest single tag
    "conservation":       "security",        # ditto
}


def _canonicalize(s) -> Optional[str]:
    if not isinstance(s, str):
        return None
    norm = s.strip().lower().replace("-", "_").replace(" ", "_")
    if norm in SCHWARTZ_VALUES:
        return norm
    if norm in _ALIASES:
        return _ALIASES[norm]
    return None


def is_schwartz_value(s) -> bool:
    """True if `s` is a recognized Schwartz value tag (after alias coercion)."""
    return _canonicalize(s) is not None


def coerce_schwartz(s) -> Optional[str]:
    """Return the canonical Schwartz tag for `s`, or None if unrecognized.
    Accepts hyphens, spaces, mixed case, and a few common aliases."""
    return _canonicalize(s)


def normalize_value_entry(entry, max_len: int = 80) -> Optional[dict]:
    """Coerce a value-list entry into the canonical `{label, schwartz}` shape.

    Accepts:
        - str: returns `{"label": str, "schwartz": None}` (legacy untagged shape)
        - dict with at least a "label" key: passes through, normalizing
          "schwartz" to a canonical tag or None

    Returns None if the entry can't be turned into something useful (no label,
    empty after stripping, wrong type).
    """
    if isinstance(entry, str):
        s = entry.strip()
        if not s:
            return None
        return {"label": s[:max_len].rstrip(), "schwartz": None}
    if isinstance(entry, dict):
        label = entry.get("label")
        if not isinstance(label, str):
            return None
        label = label.strip()
        if not label:
            return None
        return {
            "label":    label[:max_len].rstrip(),
            "schwartz": coerce_schwartz(entry.get("schwartz")),
        }
    return None
