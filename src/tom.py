"""Theory-of-Mind schema + validator + system-block formatter.

The LLM call itself lives in `user_read.py` (merged with emotion into
a single pre-generation round-trip). This module keeps the default shape,
the validator (`_validate`), and the `<theory_of_mind>` system-block formatter.
"""
import json
from typing import Optional

from parse_utils import strip_json_fences

TOM_TAG = "<theory_of_mind>"

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


def format_tom_block(tom: dict) -> str:
    """Format the ToM result as a `<theory_of_mind>` system block."""
    feeling = tom.get("feeling") or "unclear"
    avoid = tom.get("avoid") or "(no specific guidance)"
    helps = tom.get("what_helps") or "(no specific guidance)"

    return f"""
<theory_of_mind>
A read on what the user actually needs right now. Use this as a guide; do not narrate it back.

What they're feeling: {feeling}
Don't: {avoid}
Do: {helps}
</theory_of_mind>
""".strip()
