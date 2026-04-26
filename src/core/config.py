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
# State updates now happen every turn inside the merged post-exchange call
# (see post_exchange.py). The prompt itself enforces "subtle nudges only",
# so per-turn cadence does not cause drift.
KEEP_RECENT_TURNS = 8         # recent turns kept verbatim before compression

# --- EMPATHY PIPELINE ---
ENABLE_EMPATHY_PIPELINE = os.getenv("LEMON_EMPATHY", "1") not in ("0", "false", "no")
EMPATHY_RETRY_ON_FAIL = os.getenv("LEMON_EMPATHY_RETRY", "1") not in ("0", "false", "no")
MEMORY_RETRIEVAL_LIMIT = int(os.getenv("LEMON_MEMORY_LIMIT", "3"))

# --- MEMORY COMPOSITE SCORING ---
# Per-turn episodic retrieval combines four signals (lex / rec / int / emo).
# Defaults follow the ClawMem / Generative-Agents shape but skew the emotion
# weight up because empathy is the whole point. See docs/memory_architecture.md.
MEMORY_W_LEXICAL    = float(os.getenv("LEMON_MEM_W_LEXICAL",    "0.40"))
MEMORY_W_RECENCY    = float(os.getenv("LEMON_MEM_W_RECENCY",    "0.20"))
MEMORY_W_INTENSITY  = float(os.getenv("LEMON_MEM_W_INTENSITY",  "0.15"))
MEMORY_W_EMOTION    = float(os.getenv("LEMON_MEM_W_EMOTION",    "0.25"))
MEMORY_HALF_LIFE_DAYS = float(os.getenv("LEMON_MEM_HALF_LIFE_DAYS", "30"))
MEMORY_CANDIDATE_POOL = int(os.getenv("LEMON_MEM_POOL", "50"))

# --- AUTO-FACT EXTRACTION ---
ENABLE_AUTO_FACTS = os.getenv("LEMON_AUTO_FACTS", "1") not in ("0", "false", "no")
AUTO_FACTS_MAX_PER_TURN = int(os.getenv("LEMON_AUTO_FACTS_MAX", "3"))

# --- PATHS ---
# Relative paths are resolved against the project root (two levels above
# src/core/, since this module is at src/core/config.py). Without this,
# sqlite writes depend on process CWD and split the DB across multiple
# files when the server and CLI are launched from different dirs.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


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
