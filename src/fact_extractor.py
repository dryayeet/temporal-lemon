"""Automatic fact extractor.

One cheap LLM call per turn after the main reply. Reads the just-completed
exchange (user + lemon), the already-stored facts, and a few prior turns for
disambiguation. Returns a dict of NEW or UPDATED facts worth persisting so
lemon can surface them in future sessions via the `<user_facts>` system block.

Failure-tolerant: any error returns {} and the chat continues.
"""
from __future__ import annotations

import json
import re
from typing import Optional

import requests

from config import OPENROUTER_HEADERS, OPENROUTER_URL, STATE_MODEL

_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_MAX_VALUE_LEN = 200


def _build_prompt(
    user_msg: str,
    bot_reply: str,
    existing_facts: dict,
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

    return f"""
You read the most recent exchange between the user and lemon (a friendly chatbot) and decide whether it contains any durable facts about the user worth remembering across future sessions.

You are NOT replying to the user. You only produce structured facts.

Recent conversation:
{context}

Latest exchange:
Them: {user_msg}
You (lemon): {bot_reply}

Already stored facts about the user:
{known}

Extract NEW or UPDATED facts a close friend would naturally remember. Examples of what to save:
- Names (user, family, partner, pets, close friends)
- City, school, workplace, course/major, role/job
- Ongoing situations or upcoming events (exam on tuesday, wedding next month, job interview friday)
- Strong stable preferences (hates cilantro, loves dogs, plays guitar)
- Relationships (has a younger sister named Riya)

Do NOT save:
- Transient feelings or one-off moods (those are tracked elsewhere)
- Facts about lemon itself
- A fact that is already stored AND unchanged
- Low-confidence guesses
- Anything the user only implied sarcastically or hypothetically

Output rules:
- Return at most {max_new} entries.
- Keys: lowercase snake_case, letters/digits/underscore only, max 40 chars. Prefer stable semantic keys like `exam_date`, `sister_name`, `city`, `job`.
- Values: short plain strings, max 200 chars.
- If a stored fact needs updating (e.g. value changed), reuse the exact existing key so the upsert overwrites it.
- If nothing is worth saving, return an empty object `{{}}`.

Respond with ONLY the JSON object, nothing else. No prose, no markdown fences.
""".strip()


def _parse(raw: str, max_new: int) -> dict[str, str]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("fact extractor response was not a JSON object")

    out: dict[str, str] = {}
    for k, v in parsed.items():
        if not isinstance(k, str):
            continue
        key = k.strip().lower()
        if not _KEY_RE.match(key):
            continue
        if v is None:
            continue
        if not isinstance(v, str):
            v = str(v)
        value = v.strip()
        if not value:
            continue
        if len(value) > _MAX_VALUE_LEN:
            value = value[:_MAX_VALUE_LEN].rstrip()
        out[key] = value
        if len(out) >= max_new:
            break
    return out


def extract_facts(
    user_msg: str,
    bot_reply: str,
    existing_facts: Optional[dict] = None,
    recent_msgs: Optional[list[dict]] = None,
    model: Optional[str] = None,
    max_new: int = 3,
) -> dict[str, str]:
    """Return a {key: value} dict of facts to upsert. Empty dict on no-op / failure."""
    existing = existing_facts or {}
    prompt = _build_prompt(user_msg, bot_reply, existing, recent_msgs, max_new)
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=OPENROUTER_HEADERS,
            json={
                "model": model or STATE_MODEL,
                "temperature": 0.1,
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=20,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        return _parse(raw, max_new=max_new)

    except requests.HTTPError as e:
        body = getattr(e.response, "text", "")[:300]
        print(f"  [fact extractor http error: {e} | body: {body}]")
        return {}
    except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"  [fact extractor failed: {e}]")
        return {}
