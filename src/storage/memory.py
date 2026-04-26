"""Composite-scored episodic memory retrieval.

Per-turn flow:
    1. Build an FTS5 query from the user's current message
       (lowercased, stopword-filtered, OR-combined).
    2. Pull a candidate pool of past user messages from other sessions
       via SQLite FTS5 (BM25-ranked).
    3. Re-score every candidate with a weighted composite:

           score = w_lex   * normalized_bm25
                 + w_rec   * recency_decay(half_life=H days)
                 + w_int   * past_intensity
                 + w_emo   * emotion_relatedness(current, past)

    4. Return top-K candidates by composite score.

The fallback path (no FTS hits — empty/all-stopword user message, or no
shared tokens with any past message) returns the most recent user
messages from other sessions, then runs the same composite scorer
*minus* the lexical term. This means even purely-mood-driven retrieval
still works when the topical signal is silent.

The block formatter (`format_memory_block`) lives in `prompts.py`.

Empathy grounding
-----------------
Pure lexical retrieval would lose the mood-congruent recall pattern
that the empathy literature documents (Bower 1981; see also
`docs/memory_architecture.md`). We keep emotion as a strong scoring
term (default w_emo = 0.25) and use family-aware relatedness so that
sadness retrieves loneliness/grief, not only exact-label matches.
"""
from __future__ import annotations

import math
import re
from typing import Optional

from core import config
from empathy.emotion import emotion_relatedness
from core.logging_setup import get_logger, preview
from storage import db
from temporal.decay import recency_decay

log = get_logger("storage.memory")


# Small stopword list. Goal isn't grammar — it's to keep tokens like
# "the", "you", "yeah", "hai" from dominating the FTS match. Add to as
# the user's vernacular drifts.
_STOPWORDS = frozenset({
    # English
    "the", "and", "but", "for", "are", "this", "that", "with", "you", "your",
    "have", "has", "had", "was", "were", "been", "being", "from", "what",
    "when", "where", "who", "why", "how", "can", "could", "should", "would",
    "will", "shall", "may", "might", "must", "one", "two", "all", "any",
    "some", "more", "most", "other", "into", "than", "then", "them", "they",
    "their", "there", "here", "just", "only", "very", "too", "also", "now",
    "yes", "yeah", "okay", "ok", "lol", "haha", "hmm", "well", "like", "its",
    "it's", "i'm", "im", "don't", "dont", "didn't", "didnt", "won't", "wont",
    # Hinglish
    "hai", "hain", "tha", "thi", "ho", "hua", "kya", "kuch", "kar",
    "karna", "karta", "karti", "ko", "ki", "ka", "ke", "se", "mein",
    "main", "mai", "tum", "aap", "yeh", "yaar", "bhai", "bro",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize_for_fts(text: str) -> list[str]:
    """Lowercase, strip punctuation, drop short tokens + stopwords."""
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS]


def _build_fts_query(user_msg: str, max_terms: int = 20) -> Optional[str]:
    """Build an FTS5 OR-combined query from the user message. None if empty."""
    tokens = _tokenize_for_fts(user_msg)
    if not tokens:
        return None
    # Dedupe while preserving first-occurrence order, cap to avoid mega queries.
    seen: set[str] = set()
    ordered: list[str] = []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        ordered.append(t)
        if len(ordered) >= max_terms:
            break
    return " OR ".join(ordered)


_BM25_SIGMOID_SCALE = 3.0  # tuned so relevance≈3 -> 0.73, relevance≈10 -> 0.97


def _normalize_bm25(scores: list[float]) -> list[float]:
    """FTS5 BM25 is signed (smaller = more relevant). Map to [0, 1] using a
    fixed-scale sigmoid so small BM25 deltas don't blow up to full-range
    deltas (which they would under per-pool min/max).

    The sigmoid centers at "no information" (BM25=0) -> 0.5 and saturates
    smoothly toward 1.0 for strong matches. With scale=3.0:
        relevance=1   -> 0.58
        relevance=3   -> 0.73
        relevance=10  -> 0.97

    Important property: two near-identical BM25 scores yield near-identical
    normalized scores, leaving the recency / intensity / emotion terms room
    to break the tie. Min/max within a small candidate pool can't do this.
    """
    return [1.0 / (1.0 + math.exp(-(-s) / _BM25_SIGMOID_SCALE)) for s in scores]


def _composite_score(
    norm_bm25: float,
    created_at: str,
    intensity: Optional[float],
    past_emotion: Optional[str],
    current_emotion: Optional[str],
    weights: dict,
    half_life_days: float,
) -> tuple[float, dict]:
    """Combine the four signals with the configured weights. Returns
    (score, breakdown) so callers can audit retrievals."""
    rec = recency_decay(created_at or "", half_life_days)
    inten = max(0.0, min(1.0, intensity or 0.0))
    emo = emotion_relatedness(current_emotion or "", past_emotion or "")

    score = (
        weights["lex"] * norm_bm25
        + weights["rec"] * rec
        + weights["int"] * inten
        + weights["emo"] * emo
    )
    breakdown = {
        "lex": round(norm_bm25, 3),
        "rec": round(rec, 3),
        "int": round(inten, 3),
        "emo": round(emo, 3),
        "total": round(score, 3),
    }
    return score, breakdown


def relevant_memories(
    user_msg: str,
    emotion: Optional[str] = None,
    intensity: Optional[float] = None,
    current_session_id: Optional[int] = None,
    limit: Optional[int] = None,
    candidate_pool: Optional[int] = None,
    weights: Optional[dict] = None,
    half_life_days: Optional[float] = None,
    min_score: float = 0.0,
) -> list[dict]:
    """Composite-scored top-K past user messages from other sessions.

    Parameters mirror the per-turn pipeline call. `weights` and
    `half_life_days` default to the config knobs (env-tunable). Each
    returned dict has the original message columns PLUS:
      - `score`:   the composite score in [0, 1]
      - `breakdown`: {lex, rec, int, emo, total} for inspection
    """
    limit = limit if limit is not None else config.MEMORY_RETRIEVAL_LIMIT
    candidate_pool = candidate_pool if candidate_pool is not None else config.MEMORY_CANDIDATE_POOL
    weights = weights or {
        "lex": config.MEMORY_W_LEXICAL,
        "rec": config.MEMORY_W_RECENCY,
        "int": config.MEMORY_W_INTENSITY,
        "emo": config.MEMORY_W_EMOTION,
    }
    half_life_days = half_life_days if half_life_days is not None else config.MEMORY_HALF_LIFE_DAYS

    # 1. Build FTS query and pull candidates.
    fts_query = _build_fts_query(user_msg)
    used_fallback = False
    if fts_query:
        candidates = db.find_messages_by_fts(
            fts_query=fts_query,
            exclude_session_id=current_session_id,
            candidate_pool=candidate_pool,
        )
        if not candidates:
            # No lexical hits; fall through to recency/emotion-only retrieval.
            used_fallback = True
            candidates = db.find_recent_user_messages(
                exclude_session_id=current_session_id, limit=candidate_pool,
            )
    else:
        used_fallback = True
        candidates = db.find_recent_user_messages(
            exclude_session_id=current_session_id, limit=candidate_pool,
        )

    if not candidates:
        # The `remember ms=X hits=0` line in the pipeline covers this case.
        log.debug("memory_empty emotion=%s", emotion)
        return []

    # 2. Normalize BM25 scores within the pool. Fallback path has no BM25 →
    #    treat lexical term as 0 (effectively dropping it from the formula).
    if used_fallback:
        norm_bm25 = [0.0] * len(candidates)
    else:
        norm_bm25 = _normalize_bm25([c.get("bm25", 0.0) for c in candidates])

    # 3. Composite score every candidate.
    scored: list[dict] = []
    for c, nbm in zip(candidates, norm_bm25):
        score, breakdown = _composite_score(
            norm_bm25=nbm,
            created_at=c.get("created_at", ""),
            intensity=c.get("intensity"),
            past_emotion=c.get("emotion"),
            current_emotion=emotion,
            weights=weights,
            half_life_days=half_life_days,
        )
        if score < min_score:
            continue
        c2 = dict(c)
        c2["score"] = score
        c2["breakdown"] = breakdown
        scored.append(c2)

    # 4. Sort by composite score, take top-K.
    scored.sort(key=lambda r: r["score"], reverse=True)
    top = scored[:limit]

    # The pipeline's `remember ms=X hits=N` line covers the per-turn return.
    # Per-pick info stays at debug.
    log.debug(
        "memory candidates=%d returned=%d fallback=%s",
        len(candidates), len(top), used_fallback,
    )
    for r in top:
        log.debug(
            "memory_pick id=%s emotion=%s preview=%r",
            r.get("id"), r.get("emotion"), preview(r.get("content", ""), 60),
        )
    return top
