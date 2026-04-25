"""Theory-of-Mind schema + validator.

The default shape, validator, and JSON parser. The LLM call lives in
`user_read.py` (merged with emotion into a single pre-generation
round-trip). The `<theory_of_mind>` system-block formatter lives in
`prompts.py`.
"""
import json
from typing import Optional

from llm.parse_utils import strip_json_fences

DEFAULT_TOM = {
    "feeling": None,
    "avoid": None,
    "what_helps": None,
}


def _short_str(v) -> Optional[str]:
    if v is None:
        return None
    if not isinstance(v, str):
        v = str(v)
    v = v.strip()
    return v or None


def _validate(parsed: dict) -> dict:
    """Coerce an already-parsed ToM dict into the canonical schema."""
    if not isinstance(parsed, dict):
        raise ValueError("tom response was not a JSON object")
    return {
        "feeling": _short_str(parsed.get("feeling")),
        "avoid": _short_str(parsed.get("avoid")),
        "what_helps": _short_str(parsed.get("what_helps")),
    }


def _parse(raw: str) -> dict:
    """Parse a raw LLM response string (optionally fenced) into the canonical dict."""
    return _validate(json.loads(strip_json_fences(raw)))
