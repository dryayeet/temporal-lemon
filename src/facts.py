"""Format the stored user facts as a system block lemon can read."""
from typing import Mapping

FACTS_TAG = "<user_facts>"


def format_user_facts(facts: Mapping[str, str]) -> str:
    if not facts:
        return ""
    body = "\n".join(f"  {k}: {v}" for k, v in facts.items())
    return f"""
<user_facts>
Things you remember about the person you are talking to. Treat these as known background.
Do not list them out or quiz the user about them. Let them inform how you respond, naturally.

{body}
</user_facts>
""".strip()
