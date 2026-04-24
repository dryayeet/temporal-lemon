"""Theory-of-Mind schema + parser + system-block formatter.

The LLM call itself now lives in `user_read.py` (merged with emotion into
a single pre-generation round-trip). This module keeps the default shape,
the validator (`_parse`), and the `<theory_of_mind>` system-block formatter
that the pipeline injects.
"""
import json
from typing import Optional

TOM_TAG = "<theory_of_mind>"

DEFAULT_TOM = {
    "feeling": None,
    "avoid": None,
    "what_helps": None,
}


def _parse(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("tom response was not a JSON object")

    def _short_str(v) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            v = str(v)
        v = v.strip()
        return v or None

    return {
        "feeling": _short_str(parsed.get("feeling")),
        "avoid": _short_str(parsed.get("avoid")),
        "what_helps": _short_str(parsed.get("what_helps")),
    }


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
