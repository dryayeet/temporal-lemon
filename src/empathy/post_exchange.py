"""Merged post-generation bookkeeping: one LLM call covering facts + state.

Replaces the old pair of separate fact-extractor and state-updater
round-trips with a single STATE_MODEL call. The model reads the just-
completed exchange and emits a JSON object with two sub-dicts matching
the existing downstream contracts exactly.

The prompt itself lives in `prompts.build_bookkeep_prompt`.
"""
from __future__ import annotations

import json
import time
from typing import Optional

import requests

from config import OPENROUTER_HEADERS, OPENROUTER_URL, STATE_MODEL
from empathy.fact_extractor import _validate as _validate_facts
from llm.parse_utils import strip_json_fences
from logging_setup import get_logger, preview, shape_of
from prompts import build_bookkeep_prompt
from storage.state import DEFAULT_STATE, validate_state

log = get_logger("empathy.post_exchange")


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
    chosen_model = model or STATE_MODEL

    log.info(
        "event=bookkeep_call model=%s msg_len=%d reply_len=%d existing_facts=%d "
        "recent_msgs=%d max_new=%d",
        chosen_model, len(user_msg), len(bot_reply), len(existing),
        len(recent_msgs or []), max_new,
    )
    log.debug(
        "event=bookkeep_input user_msg=%r bot_reply=%r facts=%s state=%s",
        preview(user_msg), preview(bot_reply),
        shape_of(existing), shape_of(state_in),
    )

    prompt = build_bookkeep_prompt(user_msg, bot_reply, existing, state_in, recent_msgs, max_new)
    log.debug("event=bookkeep_prompt chars=%d", len(prompt))

    started = time.time()
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=OPENROUTER_HEADERS,
            json={
                "model": chosen_model,
                "temperature": 0.2,
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        elapsed_ms = int((time.time() - started) * 1000)
        log.info(
            "event=bookkeep_response status=%d elapsed_ms=%d",
            response.status_code, elapsed_ms,
        )
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage") or {}
        if usage:
            log.info(
                "event=bookkeep_usage prompt_tokens=%s completion_tokens=%s total=%s",
                usage.get("prompt_tokens"), usage.get("completion_tokens"),
                usage.get("total_tokens"),
            )

        raw = body["choices"][0]["message"]["content"]
        log.debug("event=bookkeep_raw content=%s", raw)

        parsed = json.loads(strip_json_fences(raw))
        if not isinstance(parsed, dict):
            raise ValueError("post_exchange response was not a JSON object")

        facts_sub = parsed.get("facts") if isinstance(parsed.get("facts"), dict) else {}
        state_sub = parsed.get("state") if isinstance(parsed.get("state"), dict) else {}

        new_facts = (
            _validate_facts(facts_sub, max_new, existing_keys=existing.keys())
            if facts_sub else {}
        )
        new_state = validate_state(state_sub, fallback=state_in) if state_sub else state_in

        changed_state_keys = [k for k in DEFAULT_STATE if state_in.get(k) != new_state.get(k)]
        log.info(
            "event=bookkeep_parsed new_facts=%d fact_keys=%s state_changed_keys=%s",
            len(new_facts), list(new_facts.keys()), changed_state_keys,
        )
        log.debug(
            "event=bookkeep_detail facts=%s state_before=%s state_after=%s",
            new_facts, state_in, new_state,
        )
        return new_facts, new_state

    except requests.HTTPError as e:
        body = getattr(e.response, "text", "")[:300]
        log.warning(
            "event=bookkeep_http_error error=%r body=%s — keeping existing state, no facts",
            e, body,
        )
        return {}, state_in
    except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
        log.warning(
            "event=bookkeep_failed error=%r — keeping existing state, no facts", e,
        )
        return {}, state_in
