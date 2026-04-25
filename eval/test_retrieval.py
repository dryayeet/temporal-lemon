"""Standalone retrieval-accuracy harness for lemon's memory system.

Run as:
    python eval/test_retrieval.py
    python eval/test_retrieval.py --verbose          # per-scenario detail
    python eval/test_retrieval.py --tag mood         # filter by tag
    python eval/test_retrieval.py --keep-db          # don't auto-clean fixture DB

Each scenario from `eval/fixtures.py` is loaded into an isolated temporary
SQLite DB (no contamination of `.lemon.db`). The composite retriever is
called for each probe, and we report:

  * hit@K           — was at least one expected message in the top-K?
  * recall@K        — fraction of expected messages found in the top-K
  * MRR             — mean reciprocal rank of the FIRST expected message
  * precision@K     — fraction of returned items that were expected
  * forbidden-hit-rate — fraction of probes whose top-K leaked a forbidden item
  * fallback-rate   — fraction of probes that fell back to recency-only retrieval

Reference implementations of these metrics: see Pinecone / Weaviate /
RAG-eval guides linked from `docs/memory_architecture.md`.

This script is *not* a pytest module — pytest tests live in `tests/`. Run
this when you change retrieval scoring, weights, or candidate-pool sizes
to confirm you didn't regress recall.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# Make both the project root (so `eval.fixtures` resolves) and `src/` (so
# the lemon modules resolve) importable, regardless of the cwd we're invoked from.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# IMPORTANT: redirect the lemon DB *before* importing any module that reads
# config.DB_PATH at import time (e.g. web.py reads it; we don't import that
# but storage.db does on the first connect()).
_TMP_DB = Path(tempfile.mkdtemp(prefix="lemon_eval_")) / "eval.db"
os.environ["LEMON_DB"] = str(_TMP_DB)
os.environ.setdefault("LEMON_LOG_LEVEL", "WARNING")

# Quiet OpenRouter key requirement at import — we never make HTTP calls.
os.environ.setdefault("OPENROUTER_API_KEY", "eval-noop")

from eval.fixtures import SCENARIOS  # noqa: E402

import config  # noqa: E402
from storage import db  # noqa: E402
from storage.memory import relevant_memories  # noqa: E402


# ---------- harness ----------

def _seed_scenario(scenario: dict) -> dict:
    """Insert seed messages into the DB. Returns {seed_index: db_message_id}."""
    base_now = datetime.now()
    # Group seeds by their session offset so each gets its own session row.
    sessions: dict[int, int] = {}
    seed_id_map: dict[int, int] = {}

    for idx, msg in enumerate(scenario["seed"]):
        session_offset = msg.get("session", -1)
        if session_offset not in sessions:
            sessions[session_offset] = db.start_session()

        sid = sessions[session_offset]

        # Backdate created_at by writing the row, then patching the column.
        # log_message takes its own _now() — so we update post-insert.
        msg_id = db.log_message(
            sid, msg["role"], msg["content"],
            emotion=msg.get("emotion"),
            intensity=msg.get("intensity"),
            salience=msg.get("intensity"),
        )

        days_ago = msg.get("days_ago", 0)
        backdate = (base_now - timedelta(days=days_ago)).isoformat(timespec="seconds")
        with db.connect() as c:
            c.execute("UPDATE messages SET created_at = ? WHERE id = ?",
                      (backdate, msg_id))
        seed_id_map[idx] = msg_id

    # The "current" session is whichever has session=0, else a fresh one.
    current_sid = sessions.get(0)
    if current_sid is None:
        current_sid = db.start_session()
    return {"seed_id_map": seed_id_map, "current_session_id": current_sid}


def _wipe_db() -> None:
    with db.connect() as c:
        c.execute("DELETE FROM messages")
        c.execute("DELETE FROM state_snapshots")
        c.execute("DELETE FROM sessions")
        c.execute("DELETE FROM facts")
        c.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")


def _eval_one(scenario: dict, verbose: bool = False) -> dict:
    _wipe_db()
    ctx = _seed_scenario(scenario)
    seed_id_map = ctx["seed_id_map"]
    current_sid = ctx["current_session_id"]

    probe = scenario["probe"]
    k = scenario.get("k", 3)

    results = relevant_memories(
        user_msg=probe["user_msg"],
        emotion=probe.get("emotion"),
        intensity=probe.get("intensity"),
        current_session_id=current_sid,
        limit=k,
    )

    expected_idxs = set(scenario.get("expected_in_top_k", []))
    forbidden_idxs = set(scenario.get("forbidden", []))
    expected_ids = {seed_id_map[i] for i in expected_idxs if i in seed_id_map}
    forbidden_ids = {seed_id_map[i] for i in forbidden_idxs if i in seed_id_map}

    returned_ids = [r["id"] for r in results]
    returned_id_set = set(returned_ids)

    # Metrics
    hit_at_k = bool(expected_ids & returned_id_set) if expected_ids else None
    if expected_ids:
        recall_at_k = len(expected_ids & returned_id_set) / len(expected_ids)
    else:
        recall_at_k = None
    if results:
        precision_at_k = (
            len(expected_ids & returned_id_set) / len(results) if expected_ids else None
        )
    else:
        precision_at_k = None

    # MRR — reciprocal rank of the *first* expected id appearing in returned_ids.
    rr = 0.0
    if expected_ids:
        for rank, mid in enumerate(returned_ids, start=1):
            if mid in expected_ids:
                rr = 1.0 / rank
                break

    forbidden_hit = bool(forbidden_ids & returned_id_set)

    # Negative-case test: max top score must stay below the configured ceiling.
    max_top_score = scenario.get("max_top_score")
    score_violation = False
    if max_top_score is not None and results:
        top_score = results[0].get("score", 0.0)
        score_violation = top_score > max_top_score

    fallback = bool(results) and results[0]["breakdown"]["lex"] == 0.0

    # PASS rule: hit@K must be true (or expected empty), no forbidden leak,
    # no score-ceiling violation.
    if expected_ids:
        passed = bool(hit_at_k) and not forbidden_hit and not score_violation
    else:
        passed = not forbidden_hit and not score_violation

    out = {
        "name": scenario["name"],
        "tag": scenario.get("tag", "?"),
        "k": k,
        "passed": passed,
        "hit_at_k": hit_at_k,
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
        "mrr": rr,
        "forbidden_hit": forbidden_hit,
        "score_violation": score_violation,
        "fallback": fallback,
        "results": [
            {
                "id": r["id"],
                "score": round(r.get("score", 0.0), 3),
                "breakdown": r["breakdown"],
                "emotion": r.get("emotion"),
                "content": r.get("content", "")[:80],
            }
            for r in results
        ],
        "expected_ids": sorted(expected_ids),
        "forbidden_ids": sorted(forbidden_ids),
    }

    if verbose:
        _print_scenario_detail(out)
    return out


def _print_scenario_detail(r: dict) -> None:
    flag = "PASS" if r["passed"] else "FAIL"
    print(f"\n[{flag}] {r['name']}  (tag={r['tag']}, k={r['k']})")
    if r["hit_at_k"] is not None:
        print(f"  hit@k={r['hit_at_k']}  recall@k={r['recall_at_k']:.2f}  "
              f"prec@k={(r['precision_at_k'] or 0.0):.2f}  mrr={r['mrr']:.2f}")
    print(f"  expected_ids={r['expected_ids']}  "
          f"forbidden_hit={r['forbidden_hit']}  fallback={r['fallback']}")
    for hit in r["results"]:
        marker = "*" if hit["id"] in r["expected_ids"] else (
            "X" if hit["id"] in r["forbidden_ids"] else " "
        )
        print(f"   {marker} id={hit['id']:>3}  score={hit['score']:.3f}  "
              f"emo={hit['emotion']}  | {hit['content']!r}")


def _aggregate(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    metrics_with_expected = [r for r in results if r["recall_at_k"] is not None]
    return {
        "scenarios":     n,
        "passed":        sum(1 for r in results if r["passed"]),
        "hit_at_k":      sum(1 for r in metrics_with_expected if r["hit_at_k"]) / max(1, len(metrics_with_expected)),
        "recall_at_k":   sum(r["recall_at_k"] for r in metrics_with_expected) / max(1, len(metrics_with_expected)),
        "precision_at_k": sum((r["precision_at_k"] or 0.0) for r in metrics_with_expected) / max(1, len(metrics_with_expected)),
        "mrr":           sum(r["mrr"] for r in metrics_with_expected) / max(1, len(metrics_with_expected)),
        "forbidden_hit_rate": sum(1 for r in results if r["forbidden_hit"]) / n,
        "fallback_rate":      sum(1 for r in results if r["fallback"]) / n,
    }


def _print_aggregate(per_tag: dict, overall: dict) -> None:
    print("\n" + "=" * 60)
    print(" Per-tag breakdown")
    print("=" * 60)
    for tag in sorted(per_tag):
        m = per_tag[tag]
        print(
            f"  {tag:9} | scenarios={m['scenarios']:>2}  "
            f"pass={m['passed']}/{m['scenarios']}  "
            f"hit@k={m['hit_at_k']:.2f}  recall@k={m['recall_at_k']:.2f}  "
            f"prec@k={m['precision_at_k']:.2f}  mrr={m['mrr']:.2f}  "
            f"forbidden={m['forbidden_hit_rate']:.2f}  fallback={m['fallback_rate']:.2f}"
        )
    print("\n" + "=" * 60)
    print(" Overall")
    print("=" * 60)
    print(
        f"  scenarios={overall['scenarios']}  "
        f"pass={overall['passed']}/{overall['scenarios']}  "
        f"hit@k={overall['hit_at_k']:.2f}  recall@k={overall['recall_at_k']:.2f}  "
        f"prec@k={overall['precision_at_k']:.2f}  mrr={overall['mrr']:.2f}"
    )
    print(
        f"  forbidden_hit_rate={overall['forbidden_hit_rate']:.2f}  "
        f"fallback_rate={overall['fallback_rate']:.2f}"
    )


# ---------- entry point ----------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="print per-scenario detail (top-K dump)")
    parser.add_argument("--tag", default=None,
                        help="filter scenarios by tag (topical/mood/both/negative/recency)")
    parser.add_argument("--keep-db", action="store_true",
                        help="don't auto-clean the fixture DB at exit")
    args = parser.parse_args()

    print(f"using fixture DB: {_TMP_DB}")
    print(f"weights: lex={config.MEMORY_W_LEXICAL} rec={config.MEMORY_W_RECENCY} "
          f"int={config.MEMORY_W_INTENSITY} emo={config.MEMORY_W_EMOTION}  "
          f"half_life={config.MEMORY_HALF_LIFE_DAYS}d  pool={config.MEMORY_CANDIDATE_POOL}")

    scenarios = SCENARIOS
    if args.tag:
        scenarios = [s for s in scenarios if s.get("tag") == args.tag]
        if not scenarios:
            print(f"no scenarios with tag={args.tag!r}; tags available:",
                  sorted({s.get("tag", "?") for s in SCENARIOS}))
            return 2

    results = [_eval_one(s, verbose=args.verbose) for s in scenarios]
    if not args.verbose:
        # one-line summary per scenario
        for r in results:
            flag = "PASS" if r["passed"] else "FAIL"
            print(f"  [{flag}] ({r['tag']:8}) {r['name']}")

    by_tag: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_tag[r["tag"]].append(r)
    per_tag = {tag: _aggregate(rs) for tag, rs in by_tag.items()}
    overall = _aggregate(results)
    _print_aggregate(per_tag, overall)

    if not args.keep_db:
        try:
            _TMP_DB.unlink(missing_ok=True)
            _TMP_DB.parent.rmdir()
        except OSError:
            pass

    # exit code: 0 if all passed, 1 if any failed (useful for CI)
    return 0 if overall["passed"] == overall["scenarios"] else 1


if __name__ == "__main__":
    sys.exit(main())
