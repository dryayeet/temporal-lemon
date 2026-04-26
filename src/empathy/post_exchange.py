"""Post-generation bookkeeping: facts extraction only (stages 2 + 3).

Stage 1 had this module doing both facts AND a lemon-state nudge in one
merged round-trip. Stage 2 moved the lemon-state update pre-reply (it now
flows through `empathy/user_read.py` alongside the user-state delta), so
this module shrinks to just fact extraction.

The prompt itself lives in `prompts.build_bookkeep_prompt`.
"""
from __future__ import annotations

import json
import time
from typing import Optional

import requests

from core.config import OPENROUTER_HEADERS, OPENROUTER_URL, STATE_MODEL
from core.logging_setup import get_logger, preview, shape_of
from empathy.fact_extractor import _validate as _validate_facts
from llm.parse_utils import strip_json_fences
from prompts import build_bookkeep_prompt

log = get_logger("empathy.post_exchange")


def bookkeep(
    user_msg: str,
    bot_reply: str,
    existing_facts: Optional[dict] = None,
    recent_msgs: Optional[list[dict]] = None,
    model: Optional[str] = None,
    max_new: int = 3,
) -> dict[str, str]:
    """Single STATE_MODEL round-trip. Returns a dict of new facts only.

    On any failure: returns {} so the caller can safely upsert nothing.
    """
    existing = existing_facts or {}
    chosen_model = model or STATE_MODEL

    log.debug(
        "bookkeep_input user_msg=%r reply=%r facts=%s",
        preview(user_msg), preview(bot_reply), shape_of(existing),
    )

    prompt = build_bookkeep_prompt(user_msg, bot_reply, existing, recent_msgs, max_new)

    started = time.time()
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=OPENROUTER_HEADERS,
            json={
                "model": chosen_model,
                "temperature": 0.2,
                "max_tokens": 350,  # tighter — facts only, smaller output
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        elapsed_ms = int((time.time() - started) * 1000)
        response.raise_for_status()
        body = response.json()

        raw = body["choices"][0]["message"]["content"]
        log.debug("bookkeep_raw content=%s", raw)

        parsed = json.loads(strip_json_fences(raw))
        if not isinstance(parsed, dict):
            raise ValueError("post_exchange response was not a JSON object")

        # Accept either {"facts": {...}} (legacy/merged shape) or {...} (flat)
        facts_sub = parsed.get("facts") if isinstance(parsed.get("facts"), dict) else parsed

        new_facts = (
            _validate_facts(facts_sub, max_new, existing_keys=existing.keys())
            if facts_sub else {}
        )

        log.info("bookkeep ms=%d facts=%d", elapsed_ms, len(new_facts))
        log.debug("bookkeep_detail facts=%s", new_facts)
        return new_facts

    except requests.HTTPError as e:
        body = getattr(e.response, "text", "")[:300]
        log.warning("bookkeep_http_error error=%r body=%s", e, body)
        return {}
    except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
        log.warning("bookkeep_failed error=%r", e)
        return {}
