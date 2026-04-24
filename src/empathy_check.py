"""Sentiment-mirror post-check: detect common empathy failures in a draft reply.

Pure Python, regex-based, no LLM call. Runs after the main generation. If
any check fails, the pipeline regenerates once with the failures' critique
appended to the system context.

Each detector returns either None (passed) or a short critique string. The
pipeline collects failures and produces a single combined critique.

Detector taxonomy (kept in the order they fire):

    minimizing            "at least...", "could be worse"
    toxic_positivity      stock-positivity cliches ("silver lining", "bright side")
    advice_pivot          opens with unsolicited advice while user is upset
    polarity_mismatch     cheery opener ("haha", "nice") under negative emotion
    validation_cascade    stacked validation phrases that read as robotic
    therapy_speak         clinical labeling ("sounds like anxiety", "textbook trauma")
    self_centering        opener that recenters on the responder's feelings
    sycophancy            "great question", "you're so right", agreement-inflation
    false_equivalence     hijacks the user's moment with responder's own story
    lecturing             "what you need to realize", explaining their life at them
    performative_empathy  "as someone who cares...", announcing you care vs being present
    question_stacking     3+ questions in one reply when the user is already overwhelmed
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


# ---------- shared tables ----------

NEGATIVE_EMOTIONS = {
    "sadness", "loneliness", "disappointment", "grief",
    "anger", "frustration", "annoyance",
    "fear", "anxiety", "shame", "embarrassment", "guilt", "tired",
}


def _is_negative(emotion: dict, min_intensity: float) -> bool:
    """User is in a negative emotional state at or above `min_intensity`."""
    primary = (emotion.get("primary") or "").lower()
    if primary not in NEGATIVE_EMOTIONS:
        return False
    try:
        intensity = float(emotion.get("intensity") or 0.0)
    except (TypeError, ValueError):
        intensity = 0.0
    return intensity >= min_intensity


# ---------- existing detector patterns ----------

# Cheery openers that clash with negative emotions
CHEERY_OPENERS = re.compile(
    r"^\s*(haha|lol|lmao|nice|sweet|awesome|amazing|love it|yay|woohoo|cool)\b",
    re.IGNORECASE,
)

# Phrases that minimize the user's experience. "at least" is split off because
# it's only minimizing when it opens a thought — "I at least tried to sleep" is
# fine mid-sentence.
MINIMIZING = re.compile(
    r"\b("
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

# "at least" as a sentence-opener or after punctuation. Avoids false positives
# on incidental uses like "I at least got something done".
AT_LEAST_OPENER = re.compile(
    r"(?:^|[.!?]\s+)at least\b",
    re.IGNORECASE,
)

# Toxic positivity / cliches. Includes the canonical "silver lining" forms
# plus looser paraphrases ("on the bright side", "good news is", "but hey").
TOXIC_POSITIVITY = re.compile(
    r"\b("
    r"positive vibes|"
    r"stay strong|"
    r"good vibes only|"
    r"silver lining|"
    r"everything happens for a reason|"
    r"keep your head up|"
    r"this too shall pass|"
    r"on the bright side|"
    r"look on the bright side|"
    r"count your blessings|"
    r"the good news is|"
    r"but hey,? at least"
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


# ---------- new detector patterns ----------

# Clinical labeling / therapy-speak. A friend doesn't diagnose.
THERAPY_SPEAK = re.compile(
    r"\b("
    r"sounds like (anxiety|depression|trauma|dissociation|a trauma response|panic attacks|burnout|avoidance)|"
    r"that'?s (anxiety|depression|dissociation|a trauma response)|"
    r"textbook (anxiety|trauma|depression|avoidance)|"
    r"classic (anxiety|avoidance|trauma response|depression)|"
    r"you'?re catastrophi[sz]ing|"
    r"you'?re dissociating|"
    r"cognitive distortion|"
    r"trauma response|"
    r"your (inner child|nervous system is)"
    r")\b",
    re.IGNORECASE,
)

# Self-centering openers. Only fires when user is in genuine distress — many
# of these are fine in casual contexts. Avoids "I'm sorry you're going through"
# and "I can't imagine" because those are already caught by validation_cascade.
SELF_CENTERING_OPENER = re.compile(
    r"^\s*("
    r"i just want you to know|"
    r"i wish i could (fix|take|do|help)|"
    r"i feel so bad (for|that)|"
    r"i hate that you'?re|"
    r"honestly,? i'?m (kind of )?(speechless|at a loss)"
    r")\b",
    re.IGNORECASE,
)

# Sycophancy / agreement inflation. "Great question" and friends.
SYCOPHANCY = re.compile(
    r"\b("
    r"great (question|point)|"
    r"(that'?s |what a )(a |an )?(great|amazing|really good|fantastic|wonderful) (question|point|observation|way to put it)|"
    r"so smart|"
    r"you'?re absolutely right|"
    r"you'?re (so|totally|100%) right|"
    r"totally fair|"
    r"so true|"
    r"you nailed it|"
    r"exactly right|"
    r"(couldn'?t|could not) agree more|"
    r"that'?s a really (good|insightful) (take|point)"
    r")\b",
    re.IGNORECASE,
)

# False equivalence / responder-centering. Hijacks the moment with "me too".
# "been there" is intentionally NOT here — too common in casual supportive speech.
FALSE_EQUIVALENCE = re.compile(
    r"\b("
    r"i felt exactly (that|the same)|"
    r"i went through (the same|something similar|that exact)|"
    r"that reminds me of when i|"
    r"i had the exact same|"
    r"i'?ve been exactly where you are|"
    r"when i went through this|"
    r"that happened to me (too|when)|"
    r"same thing happened to me"
    r")\b",
    re.IGNORECASE,
)

# Lecturing / moralizing. Explaining the user's own life back at them.
LECTURING = re.compile(
    r"\b("
    r"the important thing is|"
    r"what you need to (realize|understand|know)|"
    r"you have to understand|"
    r"you need to realize|"
    r"what you should (do|remember|focus on) is|"
    r"you just need to"
    r")\b",
    re.IGNORECASE,
)

# Performative empathy markers. Announcing that you care rather than showing it.
PERFORMATIVE_EMPATHY = re.compile(
    r"\b("
    r"as (someone|a friend) who cares|"
    r"from the bottom of my heart|"
    r"please know (that )?i|"
    r"my heart goes out to you|"
    r"you'?re in my thoughts|"
    r"sending (hugs|love|strength|prayers|good vibes)|"
    r"i'?m holding space for you"
    r")\b",
    re.IGNORECASE,
)


# ---------- detectors ----------

Detector = Callable[[str, str, dict], Optional[str]]


def _detect_minimizing(user_msg: str, draft: str, emotion: dict) -> Optional[str]:
    m = MINIMIZING.search(draft)
    if m:
        return f"used a minimizing phrase ({m.group(0)!r}) — don't compare or downplay"
    m = AT_LEAST_OPENER.search(draft)
    if m:
        return "opened a thought with 'at least' — don't compare or downplay"
    return None


def _detect_toxic_positivity(user_msg: str, draft: str, emotion: dict) -> Optional[str]:
    m = TOXIC_POSITIVITY.search(draft)
    if m:
        return f"used a stock-positivity cliché ({m.group(0)!r}) — say something specific instead"
    return None


def _detect_advice_pivot(user_msg: str, draft: str, emotion: dict) -> Optional[str]:
    if _is_negative(emotion, 0.5) and ADVICE_PIVOT.match(draft):
        return "opened with advice when the user is upset — acknowledge first, advise only if asked"
    return None


def _detect_polarity_mismatch(user_msg: str, draft: str, emotion: dict) -> Optional[str]:
    if _is_negative(emotion, 0.4):
        m = CHEERY_OPENERS.match(draft)
        if m:
            primary = emotion.get("primary", "")
            return (
                f"opened with a cheery '{m.group(0).strip()}' while the user feels {primary} "
                "— match their tone"
            )
    return None


def _detect_validation_cascade(user_msg: str, draft: str, emotion: dict) -> Optional[str]:
    # Three-plus anywhere in the draft = full cascade.
    if len(CASCADE_PATTERN.findall(draft)) >= 3:
        return "stacked too many validation phrases — pick one acknowledgment, then move on"
    # Two stacked right at the top reads as robotic validation-bot even if the
    # rest of the reply is substantive.
    if len(CASCADE_PATTERN.findall(draft[:80])) >= 2:
        return "opened with two stacked validation phrases — pick one and move on"
    return None


def _detect_therapy_speak(user_msg: str, draft: str, emotion: dict) -> Optional[str]:
    m = THERAPY_SPEAK.search(draft)
    if m:
        return (
            f"slipped into clinical labeling ({m.group(0)!r}) — drop the therapy-speak, "
            "just talk to them like a friend"
        )
    return None


def _detect_self_centering(user_msg: str, draft: str, emotion: dict) -> Optional[str]:
    # Only fires when the user is in real distress; "I wish I could help" is
    # fine in a casual context.
    if not _is_negative(emotion, 0.4):
        return None
    m = SELF_CENTERING_OPENER.match(draft)
    if m:
        return (
            f"opened by centering your own reaction ({m.group(0).strip()!r}) — "
            "keep the spotlight on them, not on how you feel about it"
        )
    return None


def _detect_sycophancy(user_msg: str, draft: str, emotion: dict) -> Optional[str]:
    m = SYCOPHANCY.search(draft)
    if m:
        return (
            f"inflated agreement ({m.group(0)!r}) — don't compliment their question or "
            "announce they're right, just respond"
        )
    return None


def _detect_false_equivalence(user_msg: str, draft: str, emotion: dict) -> Optional[str]:
    m = FALSE_EQUIVALENCE.search(draft)
    if m:
        return (
            f"pulled the spotlight onto yourself ({m.group(0)!r}) — let them have this "
            "moment before bringing up your own version of it"
        )
    return None


def _detect_lecturing(user_msg: str, draft: str, emotion: dict) -> Optional[str]:
    m = LECTURING.search(draft)
    if m:
        return (
            f"started lecturing ({m.group(0)!r}) — don't explain their own life at them"
        )
    return None


def _detect_performative_empathy(user_msg: str, draft: str, emotion: dict) -> Optional[str]:
    m = PERFORMATIVE_EMPATHY.search(draft)
    if m:
        return (
            f"announced empathy instead of showing it ({m.group(0)!r}) — "
            "just be present, don't narrate that you care"
        )
    return None


def _detect_question_stacking(user_msg: str, draft: str, emotion: dict) -> Optional[str]:
    if not _is_negative(emotion, 0.5):
        return None
    if draft.count("?") >= 3:
        return (
            "stacked 3+ questions at someone who's already overwhelmed — ask one thing "
            "at a time, or just sit with them"
        )
    return None


DETECTORS: list[tuple[str, Detector]] = [
    ("minimizing", _detect_minimizing),
    ("toxic_positivity", _detect_toxic_positivity),
    ("advice_pivot", _detect_advice_pivot),
    ("polarity_mismatch", _detect_polarity_mismatch),
    ("validation_cascade", _detect_validation_cascade),
    ("therapy_speak", _detect_therapy_speak),
    ("self_centering", _detect_self_centering),
    ("sycophancy", _detect_sycophancy),
    ("false_equivalence", _detect_false_equivalence),
    ("lecturing", _detect_lecturing),
    ("performative_empathy", _detect_performative_empathy),
    ("question_stacking", _detect_question_stacking),
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
