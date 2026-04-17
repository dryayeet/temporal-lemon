import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("API Key not found! Check your .env file.")

# --- MODELS ---
# Claude on OpenRouter. Strong instruction-following + supports prompt caching.
CHAT_MODEL = os.getenv("LEMON_CHAT_MODEL", "anthropic/claude-sonnet-4.6")
STATE_MODEL = os.getenv("LEMON_STATE_MODEL", "anthropic/claude-haiku-4.5")

# Set to false to send a plain string system prompt (e.g. when running on a
# non-Anthropic model that doesn't understand cache_control blocks).
ENABLE_PROMPT_CACHE = os.getenv("LEMON_PROMPT_CACHE", "1") not in ("0", "false", "no")

# --- BEHAVIOR KNOBS ---
STATE_UPDATE_EVERY = 2        # run the state updater every N exchanges
KEEP_RECENT_TURNS = 8         # recent turns kept verbatim before compression

# Streaming "human typing" pacing: base seconds/token, scaled by energy
HUMANIZE_STREAM = os.getenv("LEMON_HUMANIZE", "1") not in ("0", "false", "no")
HUMANIZE_BASE_SECONDS = 0.018       # mid-energy default
HUMANIZE_PUNCT_PAUSE = 0.18         # extra pause after . ! ? ,

# --- PATHS ---
DB_PATH = Path(os.getenv("LEMON_DB", ".lemon.db"))

# --- HTTP ---
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost/lemon",
    "X-Title": "lemon chat",
}
