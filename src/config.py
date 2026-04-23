import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("API Key not found! Check your .env file.")

# --- MODELS ---
# Claude Haiku 4.5 on OpenRouter for both the chat and the auxiliary calls.
# Cheap, fast, supports Anthropic-style prompt caching.
CHAT_MODEL = os.getenv("LEMON_CHAT_MODEL", "anthropic/claude-haiku-4.5")
STATE_MODEL = os.getenv("LEMON_STATE_MODEL", "anthropic/claude-haiku-4.5")

# Prompt caching uses Anthropic-style cache_control content blocks. OpenAI
# models don't accept that format, so we default ON only when the chat model
# is anthropic/*. Override via LEMON_PROMPT_CACHE=0 to disable.
_default_cache = "1" if CHAT_MODEL.startswith("anthropic/") else "0"
ENABLE_PROMPT_CACHE = os.getenv("LEMON_PROMPT_CACHE", _default_cache) not in ("0", "false", "no")

# --- BEHAVIOR KNOBS ---
STATE_UPDATE_EVERY = 2        # run the state updater every N exchanges
KEEP_RECENT_TURNS = 8         # recent turns kept verbatim before compression

# --- EMPATHY PIPELINE ---
ENABLE_EMPATHY_PIPELINE = os.getenv("LEMON_EMPATHY", "1") not in ("0", "false", "no")
EMPATHY_RETRY_ON_FAIL = os.getenv("LEMON_EMPATHY_RETRY", "1") not in ("0", "false", "no")
MEMORY_RETRIEVAL_LIMIT = int(os.getenv("LEMON_MEMORY_LIMIT", "3"))

# --- AUTO-FACT EXTRACTION ---
ENABLE_AUTO_FACTS = os.getenv("LEMON_AUTO_FACTS", "1") not in ("0", "false", "no")
AUTO_FACTS_MAX_PER_TURN = int(os.getenv("LEMON_AUTO_FACTS_MAX", "3"))

# Streaming "human typing" pacing: base seconds/token, scaled by energy
HUMANIZE_STREAM = os.getenv("LEMON_HUMANIZE", "1") not in ("0", "false", "no")
HUMANIZE_BASE_SECONDS = 0.018       # mid-energy default
HUMANIZE_PUNCT_PAUSE = 0.18         # extra pause after . ! ? ,

# --- PATHS ---
# Relative paths are resolved against the project root (one level above src/).
# Without this, sqlite writes depend on process CWD and split the DB across
# multiple files when the server and CLI are launched from different dirs.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _anchored(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else _PROJECT_ROOT / p


DB_PATH = _anchored(os.getenv("LEMON_DB", ".lemon.db"))

# --- HTTP ---
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost/lemon",
    "X-Title": "lemon chat",
}
