"""Merged pre-generation read: one LLM call covering emotion + theory-of-mind.

Replaces the old pair of separate emotion and ToM round-trips with a single
STATE_MODEL call. The model emits a JSON object with two sub-dicts that
match the exact shapes `emotion._validate` and `tom._validate` enforce, so
downstream (pipeline injection, db logging, trace) sees no schema change.

The prompt itself lives in `prompts.build_user_read_prompt`.
"""
from __future__ import annotations

import json
import time
from typing import Optional

import requests

from config import OPENROUTER_HEADERS, OPENROUTER_URL, STATE_MODEL
from empathy.emotion import DEFAULT_EMOTION, _validate as _validate_emotion
from empathy.tom import DEFAULT_TOM, _validate as _validate_tom
from llm.parse_utils import strip_json_fences
from logging_setup import get_logger, preview, shape_of
from prompts import build_user_read_prompt

log = get_logger("empathy.user_read")


def read_user(
    user_msg: str,
    recent_msgs: Optional[list[dict]] = None,
    model: Optional[str] = None,
) -> tuple[dict, dict]:
    """Single STATE_MODEL round-trip. Returns (emotion_dict, tom_dict).

    Falls back to (DEFAULT_EMOTION, DEFAULT_TOM) on any failure so the
    pipeline can always proceed.
    """
    chosen_model = model or STATE_MODEL
    log.debug("user_read_input user_msg=%r recent=%s",
              preview(user_msg), shape_of(recent_msgs))

    prompt = build_user_read_prompt(user_msg, recent_msgs)

    started = time.time()
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=OPENROUTER_HEADERS,
            json={
                "model": chosen_model,
                "temperature": 0.3,
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=25,
        )
        elapsed_ms = int((time.time() - started) * 1000)
        response.raise_for_status()
        body = response.json()

        raw = body["choices"][0]["message"]["content"]
        log.debug("user_read_raw content=%s", raw)

        parsed = json.loads(strip_json_fences(raw))
        if not isinstance(parsed, dict):
            raise ValueError("user_read response was not a JSON object")

        emo_sub = parsed.get("emotion") if isinstance(parsed.get("emotion"), dict) else {}
        tom_sub = parsed.get("tom") if isinstance(parsed.get("tom"), dict) else {}

        emotion = _validate_emotion(emo_sub) if emo_sub else dict(DEFAULT_EMOTION)
        tom = _validate_tom(tom_sub) if tom_sub else dict(DEFAULT_TOM)

        log.info(
            "user_read elapsed_ms=%d emotion=%s intensity=%.2f",
            elapsed_ms, emotion.get("primary"), emotion.get("intensity") or 0.0,
        )
        return emotion, tom

    except requests.HTTPError as e:
        body = getattr(e.response, "text", "")[:300]
        log.warning("user_read_http_error error=%r body=%s", e, body)
        return dict(DEFAULT_EMOTION), dict(DEFAULT_TOM)
    except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
        log.warning("user_read_failed error=%r", e)
        return dict(DEFAULT_EMOTION), dict(DEFAULT_TOM)
