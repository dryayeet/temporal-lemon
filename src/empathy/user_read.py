"""Merged pre-generation read: one LLM call covering emotion + theory-of-mind.

Replaces the old pair of separate emotion and ToM round-trips with a single
STATE_MODEL call. The model emits a JSON object with two sub-dicts that
match the exact shapes `emotion._validate` and `tom._validate` enforce, so
downstream (pipeline injection, db logging, trace) sees no schema change.

The prompt itself lives in `prompts.build_user_read_prompt`.
"""
from __future__ import annotations

import json
from typing import Optional

import requests

from config import OPENROUTER_HEADERS, OPENROUTER_URL, STATE_MODEL
from empathy.emotion import DEFAULT_EMOTION, _validate as _validate_emotion
from empathy.tom import DEFAULT_TOM, _validate as _validate_tom
from llm.parse_utils import strip_json_fences
from prompts import build_user_read_prompt


def read_user(
    user_msg: str,
    recent_msgs: Optional[list[dict]] = None,
    model: Optional[str] = None,
) -> tuple[dict, dict]:
    """Single STATE_MODEL round-trip. Returns (emotion_dict, tom_dict).

    Falls back to (DEFAULT_EMOTION, DEFAULT_TOM) on any failure so the
    pipeline can always proceed.
    """
    prompt = build_user_read_prompt(user_msg, recent_msgs)
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

        parsed = json.loads(strip_json_fences(raw))
        if not isinstance(parsed, dict):
            raise ValueError("user_read response was not a JSON object")

        emo_sub = parsed.get("emotion") if isinstance(parsed.get("emotion"), dict) else {}
        tom_sub = parsed.get("tom") if isinstance(parsed.get("tom"), dict) else {}

        emotion = _validate_emotion(emo_sub) if emo_sub else dict(DEFAULT_EMOTION)
        tom = _validate_tom(tom_sub) if tom_sub else dict(DEFAULT_TOM)
        return emotion, tom

    except requests.HTTPError as e:
        body = getattr(e.response, "text", "")[:300]
        print(f"  [user_read http error: {e} | body: {body}]")
        return dict(DEFAULT_EMOTION), dict(DEFAULT_TOM)
    except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"  [user_read failed: {e}]")
        return dict(DEFAULT_EMOTION), dict(DEFAULT_TOM)
