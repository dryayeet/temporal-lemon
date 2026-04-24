"""Fact-extraction schema + parser (key/value hygiene).

The LLM call itself now lives in `post_exchange.py` (merged with the state
updater into a single post-generation round-trip). This module keeps the
key regex, value length cap, and the validator (`_parse`) that post_exchange
delegates to.
"""
from __future__ import annotations

import json
import re

_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_MAX_VALUE_LEN = 200


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
