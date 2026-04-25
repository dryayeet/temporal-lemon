"""Merged post-generation bookkeeping: one LLM call covering facts + state.

Replaces the old pair of separate fact-extractor and state-updater
round-trips with a single STATE_MODEL call. The model reads the just-
completed exchange and emits a JSON object with two sub-dicts matching
the existing downstream contracts exactly.

The prompt itself lives in `prompts.build_bookkeep_prompt`.
"""
from __future__ import annotations

import json
from typing import Optional

import requests

from config import OPENROUTER_HEADERS, OPENROUTER_URL, STATE_MODEL
from empathy.fact_extractor import _validate as _validate_facts
from llm.parse_utils import strip_json_fences
from prompts import build_bookkeep_prompt
from storage.state import DEFAULT_STATE, validate_state


def bookkeep(
    user_msg: str,
    bot_reply: str,
    existing_facts: Optional[dict] = None,
    current_state: Optional[dict] = None,
    recent_msgs: Optional[list[dict]] = None,
    model: Optional[str] = None,
    max_new: int = 3,
) -> tuple[dict[str, str], dict]:
    """Single STATE_MODEL round-trip. Returns (new_facts, nudged_state).

    On any failure: returns ({}, current_state) so the caller can safely
    upsert nothing and keep the existing state snapshot.
    """
    existing = existing_facts or {}
    state_in = dict(current_state) if current_state else dict(DEFAULT_STATE)
    prompt = build_bookkeep_prompt(user_msg, bot_reply, existing, state_in, recent_msgs, max_new)

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=OPENROUTER_HEADERS,
            json={
                "model": model or STATE_MODEL,
                "temperature": 0.2,
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]

        parsed = json.loads(strip_json_fences(raw))
        if not isinstance(parsed, dict):
            raise ValueError("post_exchange response was not a JSON object")

        facts_sub = parsed.get("facts") if isinstance(parsed.get("facts"), dict) else {}
        state_sub = parsed.get("state") if isinstance(parsed.get("state"), dict) else {}

        new_facts = _validate_facts(facts_sub, max_new) if facts_sub else {}
        new_state = validate_state(state_sub, fallback=state_in) if state_sub else state_in
        return new_facts, new_state

    except requests.HTTPError as e:
        body = getattr(e.response, "text", "")[:300]
        print(f"  [post_exchange http error: {e} | body: {body}]")
        return {}, state_in
    except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"  [post_exchange failed: {e}]")
        return {}, state_in
