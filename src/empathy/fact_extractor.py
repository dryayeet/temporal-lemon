"""Fact-extraction schema + validator (key/value hygiene + dedup gate).

The LLM call itself lives in `post_exchange.py` (merged with the state
updater into a single post-generation round-trip). This module owns:
- The key regex and value length cap.
- `_validate`, which post_exchange delegates to.
- A reconciliation step that maps mechanically-mutated keys
  (`_final`/`_v2`/`_clarified`/`_current_X` etc.) and high-overlap
  near-duplicates back to the existing canonical key, so the model can't
  fan out a single underlying fact across N rows just by varying the
  key string.
"""
from __future__ import annotations

import json
import re
from typing import Iterable, Optional

from llm.parse_utils import strip_json_fences
from core.logging_setup import get_logger

log = get_logger("empathy.fact_extractor")

_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_MAX_VALUE_LEN = 200

# Tokens the model tacks on the END of a key when it really meant "update
# the existing fact" (e.g. `prajwal_sleep_status_final`). Strip them.
_NOISE_SUFFIXES = frozenset({
    "final", "updated", "latest", "current", "recent", "new",
    "clarified", "context", "expanded", "revised",
    "v2", "v3", "v4", "v5",
})

# Tokens the model inserts in the MIDDLE of a key as a useless modifier
# (`prajwal_current_sleep_status` vs the real `prajwal_sleep_status`).
_NOISE_INFIXES = frozenset({"current", "recent", "latest", "updated"})

# Jaccard threshold for token-set similarity. Two keys whose noise-stripped
# token sets overlap at least this much are treated as the same fact, with
# the existing key winning. 0.75 catches `prajwal_sambhav_romantic_feelings`
# vs `prajwal_sambhav_feelings` (3/4) but skips `arpit_bala_music_style` vs
# `arpit_bala_music_genre` (3/5 = 0.6) which are genuinely different.
_JACCARD_THRESHOLD = 0.75


def _strip_noise(key: str) -> str:
    """Strip recognizable noise tokens (suffixes + infixes) from a key.

    Iterative on suffixes so `_v2_final` peels both. Infix strip is
    one-pass; if every token were noise we keep the original to avoid
    returning an empty string.
    """
    tokens = key.split("_")
    while len(tokens) > 1 and tokens[-1] in _NOISE_SUFFIXES:
        tokens.pop()
    cleaned = [t for t in tokens if t not in _NOISE_INFIXES]
    if not cleaned:
        cleaned = tokens
    return "_".join(cleaned)


def _reconcile_key(new_key: str, existing_keys: Iterable[str]) -> Optional[str]:
    """Return the existing canonical key this `new_key` should map to, or
    None if `new_key` is genuinely new.

    Strategies, in order:
      1. exact match against an existing key
      2. noise-stripped form matches an existing key
      3. token-set Jaccard >= _JACCARD_THRESHOLD against any existing key
    """
    if not existing_keys:
        return None

    existing_set = set(existing_keys)
    if new_key in existing_set:
        return new_key

    stripped_new = _strip_noise(new_key)
    if stripped_new != new_key and stripped_new in existing_set:
        return stripped_new

    new_tokens = set(stripped_new.split("_"))
    if not new_tokens:
        return None

    best_key = None
    best_score = 0.0
    for ek in existing_set:
        ek_tokens = set(_strip_noise(ek).split("_"))
        if not ek_tokens:
            continue
        union = len(new_tokens | ek_tokens)
        if union == 0:
            continue
        score = len(new_tokens & ek_tokens) / union
        if score >= _JACCARD_THRESHOLD and score > best_score:
            best_key = ek
            best_score = score
    return best_key


def _validate(
    parsed: dict,
    max_new: int,
    existing_keys: Optional[Iterable[str]] = None,
) -> dict[str, str]:
    """Coerce an already-parsed facts dict into the canonical shape.

    Hygiene: keys must match `_KEY_RE`; values must be non-empty strings,
    truncated to _MAX_VALUE_LEN. Invalid entries are dropped silently.

    Dedup: when `existing_keys` is supplied, each candidate key is run
    through `_reconcile_key`. If it matches an existing key (exactly,
    after noise-stripping, or by token Jaccard), the existing key is used
    instead. This keeps the model from fanning one fact across many rows
    by mutating the key (`_final`, `_v2`, `_current`, `_clarified`, …).
    """
    if not isinstance(parsed, dict):
        raise ValueError("fact extractor response was not a JSON object")

    existing_iter = existing_keys or ()

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

        canonical = _reconcile_key(key, existing_iter)
        if canonical and canonical != key:
            log.info("fact_reconciled %s -> %s", key, canonical)
            key = canonical

        out[key] = value
        if len(out) >= max_new:
            break
    return out


def _parse(raw: str, max_new: int,
           existing_keys: Optional[Iterable[str]] = None) -> dict[str, str]:
    """Parse a raw LLM response string (optionally fenced) into a facts dict."""
    return _validate(json.loads(strip_json_fences(raw)), max_new, existing_keys)
