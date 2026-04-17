"""Sentiment-mirror post-check: detect common empathy failures in a draft reply.

Pure Python, regex-based, no LLM call. Runs after the main generation. If
any check fails, the pipeline regenerates once with the failures' critique
appended to the system context.

Each detector returns either None (passed) or a short critique string. The
pipeline collects failures and produces a single combined critique.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

# ---------- result type ----------

@dataclass
class CheckResult:
    passed: bool
    failures: list[str] = field(default_factory=list)   # human-readable labels
    critiques: list[str] = field(default_factory=list)  # one per failed detector

    @property
    def critique(self) -> str:
        """A single combined critique string suitable for prompting a regenerate."""
        if self.passed:
            return ""
        bullets = "\n".join(f"- {c}" for c in self.critiques)
        return (
            "Your previous draft had problems. Rewrite it without these patterns:\n"
            + bullets
            + "\nKeep the same intent but address the issues. Don't apologize for the previous draft."
        )


# ---------- detector helpers ----------

NEGATIVE_EMOTIONS = {
    "sadness", "loneliness", "disappointment", "grief",
    "anger", "frustration", "annoyance",
    "fear", "anxiety", "shame", "embarrassment", "guilt", "tired",
}

# Cheery openers that clash with negative emotions
CHEERY_OPENERS = re.compile(
    r"^\s*(haha|lol|lmao|nice|sweet|awesome|amazing|love it|yay|woohoo|cool)\b",
    re.IGNORECASE,
)

# Phrases that minimize the user's experience
MINIMIZING = re.compile(
    r"\b("
    r"at least|"
    r"could be worse|"
    r"everyone goes through|"
    r"that'?s nothing|"
    r"just (a|some) |"
    r"it'?s not that bad|"
    r"you'?ll be fine|"
    r"don'?t worry about it"
    r")\b",
    re.IGNORECASE,
)

# Toxic positivity / cliche
TOXIC_POSITIVITY = re.compile(
    r"\b("
    r"positive vibes|"
    r"stay strong|"
    r"good vibes only|"
    r"silver lining|"
    r"everything happens for a reason|"
    r"keep your head up|"
    r"this too shall pass"
    r")\b",
    re.IGNORECASE,
)

# Advice-pivot openers
ADVICE_PIVOT = re.compile(
    r"^\s*(you should|try (to )?|why don'?t you|have you tried|just (try|do))\b",
    re.IGNORECASE,
)

# Validation cascade signature phrases
CASCADE_PHRASES = [
    r"i hear you",
    r"that'?s so valid",
    r"your feelings are valid",
    r"that makes sense",
    r"i'?m sorry you'?re going through",
    r"that sounds really hard",
    r"i can'?t imagine",
]
CASCADE_PATTERN = re.compile("|".join(CASCADE_PHRASES), re.IGNORECASE)


# ---------- detectors ----------

Detector = Callable[[str, str, dict], Optional[str]]


def _detect_minimizing(user_msg: str, draft: str, emotion: dict) -> Optional[str]:
    if MINIMIZING.search(draft):
        match = MINIMIZING.search(draft).group(0)
        return f"used a minimizing phrase ({match!r}) — don't compare or downplay"
    return None


def _detect_toxic_positivity(user_msg: str, draft: str, emotion: dict) -> Optional[str]:
    if TOXIC_POSITIVITY.search(draft):
        match = TOXIC_POSITIVITY.search(draft).group(0)
        return f"used a stock-positivity cliché ({match!r}) — say something specific instead"
    return None


def _detect_advice_pivot(user_msg: str, draft: str, emotion: dict) -> Optional[str]:
    intensity = float(emotion.get("intensity") or 0.0)
    primary = (emotion.get("primary") or "").lower()
    if primary in NEGATIVE_EMOTIONS and intensity >= 0.5 and ADVICE_PIVOT.match(draft):
        return "opened with advice when the user is upset — acknowledge first, advise only if asked"
    return None


def _detect_polarity_mismatch(user_msg: str, draft: str, emotion: dict) -> Optional[str]:
    primary = (emotion.get("primary") or "").lower()
    intensity = float(emotion.get("intensity") or 0.0)
    if primary in NEGATIVE_EMOTIONS and intensity >= 0.4 and CHEERY_OPENERS.match(draft):
        match = CHEERY_OPENERS.match(draft).group(0).strip()
        return f"opened with a cheery '{match}' while the user feels {primary} — match their tone"
    return None


def _detect_validation_cascade(user_msg: str, draft: str, emotion: dict) -> Optional[str]:
    matches = CASCADE_PATTERN.findall(draft)
    if len(matches) >= 3:
        return "stacked too many validation phrases — pick one acknowledgment, then move on"
    return None


DETECTORS: list[tuple[str, Detector]] = [
    ("minimizing", _detect_minimizing),
    ("toxic_positivity", _detect_toxic_positivity),
    ("advice_pivot", _detect_advice_pivot),
    ("polarity_mismatch", _detect_polarity_mismatch),
    ("validation_cascade", _detect_validation_cascade),
]


def check_response(user_msg: str, draft: str, emotion: Optional[dict] = None) -> CheckResult:
    """Run every detector against `draft`. Return aggregated pass/fail + critiques."""
    emotion = emotion or {}
    failures: list[str] = []
    critiques: list[str] = []

    for label, detector in DETECTORS:
        critique = detector(user_msg, draft, emotion)
        if critique is not None:
            failures.append(label)
            critiques.append(critique)

    return CheckResult(
        passed=not failures,
        failures=failures,
        critiques=critiques,
    )
