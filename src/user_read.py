"""Merged pre-generation read: one LLM call covering emotion + theory-of-mind.

Replaces the old pair of `emotion.classify_emotion` + `tom.theory_of_mind`
round-trips with a single STATE_MODEL call. The model reads the user's
message and emits a JSON object with two sub-dicts that match the exact
shape those two functions used to return independently — so downstream
(pipeline injection, db logging, trace) sees no schema change.

Parsing delegates to `emotion._parse` and `tom._parse` via a re-dump, which
preserves every validation rule (label whitelist, intensity clamp, etc.)
without duplicating it here.
"""
from __future__ import annotations

import json
from typing import Optional

import requests

from config import OPENROUTER_HEADERS, OPENROUTER_URL, STATE_MODEL
from emotion import DEFAULT_EMOTION, EMOTION_LABELS, _parse as _parse_emotion
from tom import DEFAULT_TOM, _parse as _parse_tom


def _build_prompt(user_msg: str, recent_msgs: Optional[list[dict]]) -> str:
    context_lines: list[str] = []
    if recent_msgs:
        for m in recent_msgs[-6:]:
            role = "Them" if m["role"] == "user" else "You (lemon)"
            context_lines.append(f"{role}: {m['content']}")
    context = "\n".join(context_lines) if context_lines else "(no prior turns)"

    label_csv = ", ".join(EMOTION_LABELS)
    return f"""
You read the user's latest message and produce TWO pieces of private context for the responder. You are NOT replying to the user.

Recent conversation:
{context}

Latest user message:
"{user_msg}"

Return a JSON object with exactly two top-level keys: "emotion" and "tom".

"emotion" — a structured read of their emotional state:
  - "primary": one of [{label_csv}]
  - "intensity": float between 0.0 (very mild) and 1.0 (very strong)
  - "underlying_need": short string describing what they probably want from the next reply (e.g. "feel heard, not solved", "be distracted", "get a straight answer"), or null if unclear
  - "undertones": list of zero to three secondary emotions from the same label set

"tom" — what they actually need from the responder (be specific to THIS exchange, not generic):
  - "feeling": one sentence on what they are actually feeling, including anything they are not saying directly
  - "avoid": one specific thing the responder should NOT do (e.g. "don't jump to advice", "don't minimize with 'at least'", "don't ask another question, just sit with it")
  - "what_helps": one specific thing the responder SHOULD do to make them feel understood

Be honest. If the message is flat small-talk, "neutral" with low intensity is the right answer, and short noncommittal guidance for tom is fine. Do not over-pathologize.

Respond with ONLY the JSON object. No explanation, no markdown.
""".strip()


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def read_user(
    user_msg: str,
    recent_msgs: Optional[list[dict]] = None,
    model: Optional[str] = None,
) -> tuple[dict, dict]:
    """Single STATE_MODEL round-trip. Returns (emotion_dict, tom_dict).

    Falls back to (DEFAULT_EMOTION, DEFAULT_TOM) on any failure, so the
    pipeline can always proceed.
    """
    prompt = _build_prompt(user_msg, recent_msgs)
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=OPENROUTER_HEADERS,
            json={
                "model": model or STATE_MODEL,
                "temperature": 0.3,
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=25,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]

        parsed = json.loads(_strip_fences(raw))
        if not isinstance(parsed, dict):
            raise ValueError("user_read response was not a JSON object")

        emo_sub = parsed.get("emotion") or {}
        tom_sub = parsed.get("tom") or {}
        if not isinstance(emo_sub, dict):
            emo_sub = {}
        if not isinstance(tom_sub, dict):
            tom_sub = {}

        # Re-serialize each half and delegate to the existing validators so
        # every rule (label whitelist, intensity clamp, short-string coercion)
        # is preserved without duplication.
        emotion = _parse_emotion(json.dumps(emo_sub)) if emo_sub else dict(DEFAULT_EMOTION)
        tom = _parse_tom(json.dumps(tom_sub)) if tom_sub else dict(DEFAULT_TOM)
        return emotion, tom

    except requests.HTTPError as e:
        body = getattr(e.response, "text", "")[:300]
        print(f"  [user_read http error: {e} | body: {body}]")
        return dict(DEFAULT_EMOTION), dict(DEFAULT_TOM)
    except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"  [user_read failed: {e}]")
        return dict(DEFAULT_EMOTION), dict(DEFAULT_TOM)
