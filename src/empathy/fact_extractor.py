"""Fact-extraction schema + validator (key/value hygiene).

The LLM call itself lives in `post_exchange.py` (merged with the state
updater into a single post-generation round-trip). This module keeps the
key regex, value length cap, and the validator (`_validate`) that
post_exchange delegates to.
"""
from __future__ import annotations

import json
import re

from llm.parse_utils import strip_json_fences

_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_MAX_VALUE_LEN = 200


def _validate(parsed: dict, max_new: int) -> dict[str, str]:
    """Coerce an already-parsed facts dict into the canonical shape.

    Keys must match `_KEY_RE`; values must be short non-empty strings.
    Invalid entries are dropped silently.
    """
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


def _parse(raw: str, max_new: int) -> dict[str, str]:
    """Parse a raw LLM response string (optionally fenced) into a facts dict."""
    return _validate(json.loads(strip_json_fences(raw)), max_new)
