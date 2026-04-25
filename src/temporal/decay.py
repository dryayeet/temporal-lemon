"""Recency decay function for memory scoring.

The memory retriever weights each candidate past message by how long ago
it happened. Half-life model: a memory's recency-weight halves every
`half_life_days` days. Same shape used by Generative Agents (Park et al.
2023) and ClawMem; trivial to reason about, no special-cases.

    weight(age_days) = 0.5 ^ (age_days / half_life_days)

Examples with half_life_days=30:
    age   |  weight
    0d    |  1.000
    7d    |  0.851
    30d   |  0.500
    60d   |  0.250
    90d   |  0.125
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional


def recency_decay(
    created_at: str,
    half_life_days: float,
    now: Optional[datetime] = None,
) -> float:
    """Half-life recency weight for an ISO timestamp. Always in [0, 1]."""
    if half_life_days <= 0:
        return 1.0
    try:
        ts = datetime.fromisoformat(created_at)
    except (TypeError, ValueError):
        # Malformed timestamp → treat as "long ago, mostly faded".
        return 0.1
    n = now or datetime.now()
    age_seconds = max(0.0, (n - ts).total_seconds())
    age_days = age_seconds / 86400.0
    return 0.5 ** (age_days / half_life_days)
