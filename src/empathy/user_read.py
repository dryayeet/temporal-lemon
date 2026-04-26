"""Merged pre-generation read: one LLM call covering emotion + theory-of-mind
+ user-state delta + lemon-state delta (dyadic-state stages 1 + 2 + 3).

Single STATE_MODEL call. The model emits a JSON object with four sub-dicts
matching the shapes `emotion._validate`, `tom._validate`, and the deltas
validated by `storage.user_state.validate_delta`. With stage 2 the lemon-side
delta is computed pre-reply too, so the response generator reads a freshly-
updated lemon state.

The prompt itself lives in `prompts.build_user_read_prompt`.
"""
from __future__ import annotations

import json
import time
from typing import Optional

import requests

from core.config import OPENROUTER_HEADERS, OPENROUTER_URL, STATE_MODEL
from core.logging_setup import get_logger, preview, shape_of
from empathy.emotion import DEFAULT_EMOTION, _validate as _validate_emotion
from empathy.tom import DEFAULT_TOM, _validate as _validate_tom
from llm.parse_utils import strip_json_fences
from prompts import build_user_read_prompt
from storage.user_state import validate_delta as _validate_state_delta

log = get_logger("empathy.user_read")


def _zero_delta() -> dict:
    """Empty delta — the safe fallback when the LLM call or parse fails."""
    return _validate_state_delta({})


def _clamp_lemon_delta(delta: dict) -> dict:
    """Apply lemon-side asymmetric damping on top of the standard validator.

    Lemon's PAD nudges are clamped tighter (±0.10 vs the user's ±0.15) and
    her traits / values never move. This is the asymmetric-dynamics-with-
    symmetric-schema enforcement: same shape as user_state_delta, stricter
    bounds.
    """
    cleaned = dict(delta)
    pad = dict(cleaned.get("pad") or {})
    for k in ("pleasure", "arousal", "dominance"):
        v = float(pad.get(k, 0.0))
        pad[k] = max(-0.10, min(0.10, v))
    cleaned["pad"] = pad
    cleaned["trait_nudges"] = {}      # frozen
    cleaned["value_add"] = []          # frozen
    return cleaned


def read_user(
    user_msg: str,
    recent_msgs: Optional[list[dict]] = None,
    current_user_state: Optional[dict] = None,
    current_lemon_state: Optional[dict] = None,
    model: Optional[str] = None,
) -> tuple[dict, dict, dict, dict]:
    """Single STATE_MODEL round-trip. Returns
    (emotion, tom, user_state_delta, lemon_state_delta).

    Falls back to safe defaults on any failure so the pipeline can always
    proceed and a bad LLM response can never poison the persisted state of
    either agent.
    """
    chosen_model = model or STATE_MODEL
    log.debug(
        "user_read_input user_msg=%r recent=%s user=%s lemon=%s",
        preview(user_msg), shape_of(recent_msgs),
        shape_of(current_user_state), shape_of(current_lemon_state),
    )

    prompt = build_user_read_prompt(
        user_msg, recent_msgs,
        current_user_state=current_user_state,
        current_lemon_state=current_lemon_state,
    )

    started = time.time()
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=OPENROUTER_HEADERS,
            json={
                "model": chosen_model,
                "temperature": 0.3,
                "max_tokens": 900,  # bumped — four sub-objects now
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
        user_delta_sub = parsed.get("user_state_delta") if isinstance(parsed.get("user_state_delta"), dict) else {}
        lemon_delta_sub = parsed.get("lemon_state_delta") if isinstance(parsed.get("lemon_state_delta"), dict) else {}

        emotion = _validate_emotion(emo_sub) if emo_sub else dict(DEFAULT_EMOTION)
        tom = _validate_tom(tom_sub) if tom_sub else dict(DEFAULT_TOM)
        user_delta = _validate_state_delta(user_delta_sub)
        lemon_delta = _clamp_lemon_delta(_validate_state_delta(lemon_delta_sub))

        log.info(
            "read ms=%d emo=%s i=%.2f user=%s lemon=%s",
            elapsed_ms, emotion.get("primary"), emotion.get("intensity") or 0.0,
            user_delta.get("mood_label") or "-",
            lemon_delta.get("mood_label") or "-",
        )
        return emotion, tom, user_delta, lemon_delta

    except requests.HTTPError as e:
        body = getattr(e.response, "text", "")[:300]
        log.warning("user_read_http_error error=%r body=%s", e, body)
        return dict(DEFAULT_EMOTION), dict(DEFAULT_TOM), _zero_delta(), _zero_delta()
    except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
        log.warning("user_read_failed error=%r", e)
        return dict(DEFAULT_EMOTION), dict(DEFAULT_TOM), _zero_delta(), _zero_delta()
