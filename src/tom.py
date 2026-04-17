"""Theory-of-Mind side pass.

Second cheap LLM call per turn. Given the user's message + the emotion
classifier's reading + recent context, infer (a) what they actually feel,
(b) what they don't want to hear, (c) what would make them feel understood.

Cheap, structured, advisory — never produces user-facing text. The result
is injected as a `<theory_of_mind>` system block.
"""
import json
from typing import Optional

import requests

from config import OPENROUTER_HEADERS, OPENROUTER_URL, STATE_MODEL

TOM_TAG = "<theory_of_mind>"

DEFAULT_TOM = {
    "feeling": None,
    "avoid": None,
    "what_helps": None,
}


def _build_prompt(user_msg: str, emotion: Optional[dict],
                  recent_msgs: Optional[list[dict]]) -> str:
    context_lines = []
    if recent_msgs:
        for m in recent_msgs[-6:]:
            role = "Them" if m["role"] == "user" else "You (lemon)"
            context_lines.append(f"{role}: {m['content']}")
    context = "\n".join(context_lines) if context_lines else "(no prior turns)"

    if emotion:
        emo_line = (
            f"Emotion classifier says: primary={emotion.get('primary')}, "
            f"intensity={emotion.get('intensity', 0):.2f}, "
            f"undertones={emotion.get('undertones') or []}, "
            f"underlying_need={emotion.get('underlying_need')}"
        )
    else:
        emo_line = "Emotion classifier did not run."

    return f"""
You are a quiet observer thinking about what the user actually needs from the next reply. You are NOT replying to them. Your output is private context for the responder.

Recent conversation:
{context}

Latest user message:
"{user_msg}"

{emo_line}

Return a JSON object with exactly these three keys, each a short string (one sentence, plain English, no jargon):
- "feeling": what they are actually feeling, including anything they are not saying directly
- "avoid": one specific thing the responder should NOT do (e.g. "don't jump to advice", "don't minimize with 'at least'", "don't ask another question, just sit with it")
- "what_helps": one specific thing the responder SHOULD do to make them feel understood

Be concrete and specific to this exchange, not generic. If the message is genuinely flat ("ok", "lol"), short noncommittal answers are fine.

Respond with ONLY the JSON object. No explanation, no markdown.
""".strip()


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


def theory_of_mind(
    user_msg: str,
    emotion: Optional[dict] = None,
    recent_msgs: Optional[list[dict]] = None,
    model: Optional[str] = None,
) -> dict:
    """Return a 3-key dict (feeling, avoid, what_helps). Falls back to DEFAULT_TOM on any failure."""
    prompt = _build_prompt(user_msg, emotion, recent_msgs)
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=OPENROUTER_HEADERS,
            json={
                "model": model or STATE_MODEL,
                "temperature": 0.4,
                "max_tokens": 350,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=25,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        return _parse(raw)

    except requests.HTTPError as e:
        body = getattr(e.response, "text", "")[:300]
        print(f"  [tom http error: {e} | body: {body}]")
        return dict(DEFAULT_TOM)
    except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"  [tom failed: {e}]")
        return dict(DEFAULT_TOM)


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
