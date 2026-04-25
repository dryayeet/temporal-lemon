"""Curated retrieval-evaluation fixtures.

Each scenario is a small synthetic conversation history (`seed`) plus a
probe (current user message + classifier output) and a list of message
indices that *should* appear in the top-K retrieval result.

Scenario shape
--------------
{
    "name":     "human-readable label",
    "tag":      "topical | mood | both | negative | recency",
    "k":        3,                          # top-K size to evaluate
    "seed": [                                # past messages, oldest first
        {"days_ago": 14, "session": -3, "role": "user",
         "content": "...", "emotion": "anxiety", "intensity": 0.6},
        {"days_ago": 14, "session": -3, "role": "assistant", "content": "..."},
        ...
    ],
    "probe": {
        "user_msg": "...",
        "emotion":  "disappointment",
        "intensity": 0.6,
    },
    # which seed-indices SHOULD appear in top-K (any order)
    "expected_in_top_k": [0],
    # optional: which seed-indices SHOULD ABSOLUTELY NOT appear
    "forbidden": [],
}

Tag meanings
------------
  topical  — the right answer shares topic/keywords with the probe
  mood     — the right answer shares the probe's emotion family but not topic
  both     — topical AND mood-aligned
  negative — there's no right answer; system should retrieve nothing high-confidence
  recency  — multiple matches; the most recent should rank higher

Tags are aggregated in the report so we can see per-category accuracy.
"""
from __future__ import annotations

SCENARIOS: list[dict] = [

    # 1. Pure topical — neutral mood, exam comes up again
    {
        "name": "neutral probe re-surfaces topical past",
        "tag": "topical",
        "k": 3,
        "seed": [
            {"days_ago": 7, "session": -2, "role": "user",
             "content": "i have an exam on tuesday next week",
             "emotion": "anxiety", "intensity": 0.5},
            {"days_ago": 7, "session": -2, "role": "assistant",
             "content": "oh that's soon, you prepping?"},
            {"days_ago": 5, "session": -1, "role": "user",
             "content": "ate biryani for lunch today",
             "emotion": "neutral", "intensity": 0.2},
            {"days_ago": 5, "session": -1, "role": "user",
             "content": "watching a movie with my flatmate",
             "emotion": "neutral", "intensity": 0.2},
        ],
        "probe": {
            "user_msg": "the exam went pretty bad",
            "emotion": "disappointment",
            "intensity": 0.6,
        },
        "expected_in_top_k": [0],
        "forbidden": [],
    },

    # 2. Mood-only — no topical overlap, but mood family matches
    {
        "name": "mood-aligned recall when topic is different",
        "tag": "mood",
        "k": 3,
        "seed": [
            {"days_ago": 20, "session": -3, "role": "user",
             "content": "i feel so alone these days",
             "emotion": "loneliness", "intensity": 0.7},
            {"days_ago": 10, "session": -2, "role": "user",
             "content": "spent the weekend hiking",
             "emotion": "joy", "intensity": 0.6},
            {"days_ago": 3, "session": -1, "role": "user",
             "content": "got promoted at work today",
             "emotion": "excitement", "intensity": 0.8},
        ],
        "probe": {
            "user_msg": "feeling kind of low tonight",
            "emotion": "sadness",
            "intensity": 0.6,
        },
        "expected_in_top_k": [0],  # the loneliness msg is family-related
        "forbidden": [1, 2],       # joy msgs should not surface
    },

    # 3. Topical AND mood-aligned — easiest case
    {
        "name": "topic + mood both align",
        "tag": "both",
        "k": 3,
        "seed": [
            {"days_ago": 30, "session": -4, "role": "user",
             "content": "my dog passed away last night",
             "emotion": "grief", "intensity": 0.95},
            {"days_ago": 15, "session": -2, "role": "user",
             "content": "got a new haircut",
             "emotion": "neutral", "intensity": 0.3},
            {"days_ago": 2, "session": -1, "role": "user",
             "content": "still missing my dog so much",
             "emotion": "grief", "intensity": 0.8},
        ],
        "probe": {
            "user_msg": "thinking about my dog again",
            "emotion": "sadness",
            "intensity": 0.7,
        },
        "expected_in_top_k": [0, 2],
        "forbidden": [1],
    },

    # 4. Negative case — probe is about something never mentioned
    {
        "name": "no relevant memory exists — must not surface garbage",
        "tag": "negative",
        "k": 3,
        "seed": [
            {"days_ago": 5, "session": -1, "role": "user",
             "content": "playing badminton in the evenings now",
             "emotion": "joy", "intensity": 0.5},
            {"days_ago": 5, "session": -1, "role": "user",
             "content": "trying out a new chai recipe",
             "emotion": "neutral", "intensity": 0.3},
        ],
        "probe": {
            "user_msg": "thinking about getting into pottery",
            "emotion": "curious",
            "intensity": 0.4,
        },
        # No exact match. Fallback retrieval will return *something* — but the
        # top score should be low. We assert no message is above 0.55 here.
        "expected_in_top_k": [],
        "forbidden": [],
        "max_top_score": 0.55,
    },

    # 5. Recency — multiple equal-emotion matches, recent should win
    {
        "name": "recency tiebreaker among equal-emotion matches",
        "tag": "recency",
        "k": 1,
        "seed": [
            {"days_ago": 90, "session": -5, "role": "user",
             "content": "got into a fight with my brother",
             "emotion": "anger", "intensity": 0.6},
            {"days_ago": 30, "session": -3, "role": "user",
             "content": "fought with my brother again",
             "emotion": "anger", "intensity": 0.6},
            {"days_ago": 3, "session": -1, "role": "user",
             "content": "another fight with my brother today",
             "emotion": "anger", "intensity": 0.6},
        ],
        "probe": {
            "user_msg": "fight with my brother",
            "emotion": "anger",
            "intensity": 0.6,
        },
        "expected_in_top_k": [2],
        "forbidden": [],
    },

    # 6. Cross-session exclusion — current-session msg must not be retrieved
    {
        "name": "current-session messages are excluded",
        "tag": "topical",
        "k": 3,
        "seed": [
            {"days_ago": 20, "session": -2, "role": "user",
             "content": "starting a new job at a fintech",
             "emotion": "excitement", "intensity": 0.7},
            # this lands in CURRENT session (session=0) and must NOT come back
            {"days_ago": 0, "session": 0, "role": "user",
             "content": "first day at the fintech today",
             "emotion": "excitement", "intensity": 0.7},
        ],
        "probe": {
            "user_msg": "how is the fintech going",
            "emotion": "curious",
            "intensity": 0.3,
        },
        "expected_in_top_k": [0],
        "forbidden": [1],
    },

    # 7. Stem-aware match — porter stemming should let "sleeping" match "sleep"
    {
        "name": "stemming connects sleep / sleeping / slept",
        "tag": "topical",
        "k": 3,
        "seed": [
            {"days_ago": 4, "session": -1, "role": "user",
             "content": "haven't been sleeping well",
             "emotion": "tired", "intensity": 0.6},
            {"days_ago": 4, "session": -1, "role": "user",
             "content": "made some progress on the side project",
             "emotion": "joy", "intensity": 0.4},
        ],
        "probe": {
            "user_msg": "did you sleep okay last night",
            "emotion": "neutral",
            "intensity": 0.2,
        },
        "expected_in_top_k": [0],
        "forbidden": [],
    },

    # 8. High-intensity past memory beats low-intensity tie
    {
        "name": "intensity tiebreaks comparable matches",
        "tag": "both",
        "k": 1,
        "seed": [
            {"days_ago": 14, "session": -2, "role": "user",
             "content": "anxious about the talk i'm giving",
             "emotion": "anxiety", "intensity": 0.3},
            {"days_ago": 14, "session": -2, "role": "user",
             "content": "really anxious about the talk i'm giving",
             "emotion": "anxiety", "intensity": 0.85},
        ],
        "probe": {
            "user_msg": "anxious about the talk",
            "emotion": "anxiety",
            "intensity": 0.7,
        },
        "expected_in_top_k": [1],
        "forbidden": [],
    },

    # 9. Cross-family different emotion shouldn't beat topical match
    {
        "name": "topical match outranks unrelated-mood match",
        "tag": "topical",
        "k": 1,
        "seed": [
            {"days_ago": 6, "session": -1, "role": "user",
             "content": "planning a trip to goa next month",
             "emotion": "joy", "intensity": 0.6},
            {"days_ago": 30, "session": -3, "role": "user",
             "content": "completely overwhelmed with work right now",
             "emotion": "anxiety", "intensity": 0.8},
        ],
        "probe": {
            "user_msg": "thinking about that goa trip",
            "emotion": "joy",
            "intensity": 0.5,
        },
        "expected_in_top_k": [0],
        "forbidden": [1],
    },

    # 10. Stale memory (very old) should still surface if mood + topic align
    {
        "name": "old but salient memory still retrievable",
        "tag": "both",
        "k": 3,
        "seed": [
            {"days_ago": 180, "session": -10, "role": "user",
             "content": "my grandmother passed away last week",
             "emotion": "grief", "intensity": 0.95},
            {"days_ago": 5, "session": -1, "role": "user",
             "content": "had pasta for dinner",
             "emotion": "neutral", "intensity": 0.2},
        ],
        "probe": {
            "user_msg": "missing my grandmother today",
            "emotion": "grief",
            "intensity": 0.7,
        },
        "expected_in_top_k": [0],
        "forbidden": [1],
    },
]
