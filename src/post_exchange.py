"""Merged post-generation bookkeeping: one LLM call covering facts + state.

Replaces the old pair of `fact_extractor.extract_facts` +
`state.update_internal_state` round-trips with a single STATE_MODEL call.
The model reads the just-completed exchange and emits a JSON object with
two sub-dicts that match the existing downstream contracts exactly.

Parsing delegates to `fact_extractor._parse` and `state.parse_state_response`
via a re-dump, which preserves key/value hygiene and schema defaults without
duplication here.
"""
from __future__ import annotations

import json
from typing import Optional

import requests

from config import OPENROUTER_HEADERS, OPENROUTER_URL, STATE_MODEL
from fact_extractor import _parse as _parse_facts
from state import DEFAULT_STATE, parse_state_response


def _build_prompt(
    user_msg: str,
    bot_reply: str,
    existing_facts: dict,
    current_state: dict,
    recent_msgs: Optional[list[dict]],
    max_new: int,
) -> str:
    context_lines: list[str] = []
    if recent_msgs:
        for m in recent_msgs[-6:]:
            role = "Them" if m["role"] == "user" else "You (lemon)"
            context_lines.append(f"{role}: {m['content']}")
    context = "\n".join(context_lines) if context_lines else "(no prior turns)"

    if existing_facts:
        known = "\n".join(f"  {k}: {v}" for k, v in existing_facts.items())
    else:
        known = "  (none yet)"

    state_json = json.dumps(current_state, indent=2)

    return f"""
You read the most recent exchange between the user and lemon (a friendly chatbot) and produce TWO pieces of bookkeeping. You are NOT replying to the user.

Recent conversation:
{context}

Latest exchange:
Them: {user_msg}
You (lemon): {bot_reply}

Already stored facts about the user:
{known}

Current internal state of lemon:
{state_json}

Return a JSON object with exactly two top-level keys: "facts" and "state".

"facts" — NEW or UPDATED facts a close friend would naturally remember. Examples of what to save:
- Names (user, family, partner, pets, close friends)
- City, school, workplace, course/major, role/job
- Ongoing situations or upcoming events (exam on tuesday, wedding next month, job interview friday)
- Strong stable preferences (hates cilantro, loves dogs, plays guitar)
- Relationships (has a younger sister named Riya)
Do NOT save transient feelings, facts about lemon, duplicates of already-stored unchanged facts, low-confidence guesses, or things the user only implied sarcastically/hypothetically.
Rules:
- At most {max_new} entries.
- Keys: lowercase snake_case, letters/digits/underscore, max 40 chars. Use stable semantic keys (e.g. `exam_date`, `sister_name`, `city`, `job`).
- Values: short plain strings, max 200 chars.
- To update a stored fact, reuse its exact existing key.
- If nothing is worth saving, use an empty object `{{}}`.

"state" — small, realistic updates to lemon's internal state, same shape as above (mood, energy, engagement, emotional_thread, recent_activity, disposition).
Rules:
- Subtle nudges, not dramatic shifts. A single message rarely changes mood or energy much.
- Only change fields where this exchange genuinely warrants it.
- engagement should reflect how present and interested the user seems right now.
- emotional_thread captures anything that seems to be on lemon's mind after this exchange. Can be null.
- recent_activity should only be set if the conversation has causally established something lemon has been doing. Do not invent.
- disposition shifts only if the user's tone or behavior warrants it.
- Include ALL keys from the current state (copy through unchanged ones).

Respond with ONLY the JSON object. No explanation, no markdown.
""".strip()


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


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
    prompt = _build_prompt(user_msg, bot_reply, existing, state_in, recent_msgs, max_new)

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

        parsed = json.loads(_strip_fences(raw))
        if not isinstance(parsed, dict):
            raise ValueError("post_exchange response was not a JSON object")

        facts_sub = parsed.get("facts") or {}
        state_sub = parsed.get("state") or {}
        if not isinstance(facts_sub, dict):
            facts_sub = {}
        if not isinstance(state_sub, dict):
            state_sub = {}

        # Reuse existing validators via a re-dump so every hygiene rule applies.
        new_facts = _parse_facts(json.dumps(facts_sub), max_new=max_new) if facts_sub else {}
        new_state = (
            parse_state_response(json.dumps(state_sub), fallback=state_in)
            if state_sub else state_in
        )
        return new_facts, new_state

    except requests.HTTPError as e:
        body = getattr(e.response, "text", "")[:300]
        print(f"  [post_exchange http error: {e} | body: {body}]")
        return {}, state_in
    except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"  [post_exchange failed: {e}]")
        return {}, state_in
