# Memory architecture

> Status: live as of 2026-04-27. Tier 2 (composite scoring) shipped 2026-04-26;
> facts moved to a facts-only post-exchange call as part of the dyadic-state
> stage 2 work on 2026-04-27.

This doc covers how lemon remembers things — what's persisted, what's
surfaced into each turn's prompt, and how retrieval is scored. There are
**three tiers**, each with a different update cadence and selection rule.

```
Tier 1 — facts (always-on)        ↓ every turn
Tier 2 — episodic memories         ↓ scored, top-K per turn
Tier 3 — working history           ↓ recent verbatim + folded summary
```

The tonic state for both lemon and the user (the three-layer Big 5 +
adaptations + PAD object covered in `docs/dyadic_state.md`) sits *alongside*
this memory architecture, not inside it. Persistent state and persistent
memories are different objects with different update sites:

- **State**: nudged pre-reply by the merged user_read pass, persisted in
  `lemon_state_snapshots` and `user_state_snapshots`.
- **Facts**: extracted post-reply by `bookkeep` (now facts-only), persisted
  in the `facts` table.
- **Episodic memories**: every user message gets a phasic emotion tag at log
  time; the FTS5+composite retriever pulls from the `messages` table.

## Tier 1 — Facts (always-on, no retrieval)

Durable key/value bookkeeping. Names, ongoing situations, stable preferences,
relationships. Updated by the post-reply `bookkeep` LLM call
(`empathy/post_exchange.py`) — which is now facts-only after stage 2 of the
dyadic-state work moved the state nudge pre-reply. Stored in the SQLite
`facts` table.

Every turn dumps **all** facts into the system stack as the `<user_facts>`
block (`prompts.format_user_facts`, refreshed by `session_context.refresh_base_blocks`).

There's no per-turn filtering — facts are small (typically <200 entries) and
cheap to ship. The dedup gate (`empathy/fact_extractor._reconcile_key`) keeps
the LLM from inventing key mutations.

## Tier 2 — Episodic memories (scored retrieval)

Past user messages, retrieved per turn by composite scoring. This is the tier
that recently changed: it used to be an exact-emotion-label filter with a
neutral early-out. Now it's **multi-signal scoring with emotion as a boost,
not a filter**.

### Composite scoring formula

For each candidate past user message:

```
score = w_lex   · sigmoid(-bm25 / 3.0)                    # topical relevance
      + w_rec   · 0.5 ^ (age_days / half_life_days)       # recency decay
      + w_int   · clamp(intensity, 0, 1)                  # past-message salience
      + w_emo   · emotion_relatedness(current, past)      # mood-congruence
```

Defaults (`src/config.py`, env-tunable):

| weight              | env var                          | default | what it values         |
| ------------------- | -------------------------------- | ------- | ---------------------- |
| `w_lex`             | `LEMON_MEM_W_LEXICAL`            | 0.40    | topic / keyword overlap |
| `w_rec`             | `LEMON_MEM_W_RECENCY`            | 0.20    | how recent             |
| `w_int`             | `LEMON_MEM_W_INTENSITY`          | 0.15    | how intensely felt     |
| `w_emo`             | `LEMON_MEM_W_EMOTION`            | 0.25    | mood-congruence        |
| `half_life_days`    | `LEMON_MEM_HALF_LIFE_DAYS`       | 30      | recency decay rate     |
| `candidate_pool`    | `LEMON_MEM_POOL`                 | 50      | size of the FTS pool   |
| `top_K returned`    | `LEMON_MEMORY_LIMIT`             | 3       | injected into the prompt |

The lexical weight is the largest because **topical relevance is the signal
the previous design was missing**. Emotion at 0.25 keeps mood-congruent recall
load-bearing without making it the only retrieval cue.

### Sigmoid lexical normalization

FTS5 BM25 returns signed scores (smaller = more relevant). We map BM25 → [0,1]
via `1 / (1 + exp(-(-bm25) / 3.0))` — a fixed-scale sigmoid, **not**
per-pool min/max. This matters: with min/max, two near-identical BM25 values
would spread to 0.0 and 1.0, swamping the other terms; with the sigmoid they
stay close, leaving room for recency / intensity / emotion to break the tie.

### Emotion relatedness — family-aware, not exact-only

`empathy/emotion.emotion_relatedness(a, b)`:

| relationship                                  | weight |
| --------------------------------------------- | ------ |
| same exact label (and not `neutral`)          | 1.0    |
| different label, same family                  | 0.5    |
| different family, or either is `neutral`      | 0.0    |

Family map (`empathy/emotion.EMOTION_FAMILIES`). Sad / anger / fear follow
Shaver et al.'s prototype hierarchy; the self-conscious cluster follows
Tracy & Robins (shared self-representation substrate, which is why pride
groups with shame/guilt despite the opposite valence); positive is PANAS
high-valence; low-arousal and exploratory are kept separate from the basic
emotions so they don't confer mood-congruence on sad-cluster memories.

| family            | labels                                                |
| ----------------- | ----------------------------------------------------- |
| `positive`        | joy, excitement, love, gratitude, relief, amused      |
| `sad`             | sadness, loneliness, disappointment, grief            |
| `anger`           | anger, frustration, annoyance                         |
| `fear`            | fear, anxiety, confusion                              |
| `self_conscious`  | shame, embarrassment, guilt, pride                    |
| `low_arousal`     | tired                                                 |
| `exploratory`     | curious                                               |
| `neutral`         | neutral                                               |

Why neutral never scores: every turn is mostly neutral, so neutral×neutral
matching would dominate the bonus and defeat its purpose.

### FTS5 setup

The `messages_fts` virtual table is an external-content FTS5 index over
`messages.content`, with the porter+unicode61 tokenizer (so `sleep`,
`sleeping`, `slept` all match). Three triggers (`messages_ai`, `messages_ad`,
`messages_au`) keep it in sync with inserts/deletes/updates.

A migration step in `connect()` (`_rebuild_fts_if_needed`) backfills the
index from existing rows when an old DB is opened the first time.

The FTS query is built in `storage/memory._build_fts_query`: lowercase,
strip punctuation, drop stopwords (English + a small Hinglish set), drop
short tokens (<3 chars), dedupe, OR-combine, cap at 20 terms.

### Fallback path

When the user message yields no FTS query (all-stopwords) or returns no hits,
the retriever falls back to "most recent N user messages from other sessions"
and runs the **same composite scorer** on them with `lex=0`. So even purely
mood-driven retrieval still works when the topical signal is silent — the
breakdown just shows `lex: 0.0`.

The eval harness reports `fallback_rate` so we can monitor how often this
path fires (negative-case scenarios are the only ones that should).

### Per-turn flow

```
user_msg                                          (e.g. "the exam went bad")
   │
   ▼
build_fts_query → "exam OR went OR bad"
   │
   ▼
db.find_messages_by_fts(query, exclude=current_session, pool=50)
   │
   ▼  (or fallback to find_recent_user_messages on miss)
candidate pool of past user messages
   │
   ▼
for each candidate:
    composite_score(bm25, age, intensity, past_emotion vs current_emotion)
   │
   ▼
sort by composite score, take top-K
   │
   ▼
format_memory_block(top_K) → <emotional_memory> system block
   │
   ▼
inject into history → drafting LLM call
```

Logged at INFO via `lemon.storage.memory`:

* `event=memory_retrieved` — count, fallback flag, top scores
* `event=memory_pick` (DEBUG) — per-pick id / breakdown / preview

## Tier 3 — Working history (within-session)

Recent turns kept verbatim, older ones folded into a single
`<earlier_conversation>` summary block by `prompt_stack.compress_history`.
Configured by `KEEP_RECENT_TURNS` (default 8). Unchanged by this work.

# Evaluation

Standalone harness at `eval/test_retrieval.py`. Spins up an isolated SQLite
DB in `tempfile.mkdtemp`, seeds 10 curated scenarios, runs the composite
retriever, computes:

| metric              | what it measures                                                     |
| ------------------- | -------------------------------------------------------------------- |
| `hit@K`             | did at least one expected memory appear in the top-K?                 |
| `recall@K`          | fraction of expected memories found in top-K                          |
| `precision@K`       | fraction of returned items that were expected                         |
| `MRR`               | mean reciprocal rank of the first expected memory                     |
| `forbidden_hit_rate`| fraction of probes whose top-K leaked a forbidden item                |
| `fallback_rate`     | fraction of probes that fell back to recency-only retrieval           |

Scenario tags (each tests a specific axis):

| tag        | tests                                                              |
| ---------- | ------------------------------------------------------------------ |
| `topical`  | keyword/topic overlap surfaces the right past message              |
| `mood`     | mood-congruence surfaces the right past message even with no topic |
| `both`     | both signals align (easiest case)                                  |
| `negative` | nothing relevant exists; system should not surface garbage         |
| `recency`  | among equally-relevant past messages, the recent one ranks first   |

Run:

```bash
python eval/test_retrieval.py              # one-line summary per scenario + aggregate
python eval/test_retrieval.py --verbose    # full top-K dump with score breakdowns
python eval/test_retrieval.py --tag mood   # only mood scenarios
```

Exit code is 0 if all pass, 1 if any fail — drop into CI when you want a
regression gate.

Current score (10 scenarios): **hit@K = 1.00, recall@K = 1.00, precision@K = 1.00, MRR = 1.00, forbidden_hit_rate = 0.00**.

# Knobs you can turn

All composite-scoring weights and the half-life are env-tunable without
code changes — useful for A/B'ing weight changes against the eval harness.
Examples:

```bash
# Try a more topical-leaning scorer
LEMON_MEM_W_LEXICAL=0.55 LEMON_MEM_W_EMOTION=0.10 python eval/test_retrieval.py

# Faster decay (memories age out at 14 days instead of 30)
LEMON_MEM_HALF_LIFE_DAYS=14 python eval/test_retrieval.py

# Larger candidate pool (more candidates to score, slower)
LEMON_MEM_POOL=200 python eval/test_retrieval.py
```

# Future work

Things explicitly **not** in this iteration, in rough order of return:

1. **Semantic embeddings.** BM25 catches `sleep / sleeping / slept` via porter
   stemming but not paraphrases (`exhausted` ↔ `wiped out`). Next step is to
   add an embedding signal (BGE-M3 or similar) and combine with BM25 via
   reciprocal rank fusion (RRF). This is the Mem0/ClawMem pattern.
2. **Reflection / consolidation.** Every past message stays as a row forever.
   At ~10K rows the FTS pool will start surfacing noise. Pattern: periodic
   LLM summarization (post-session or every N turns) into 1-3 sentence
   "shape of what was discussed" rollups, indexed alongside raw messages
   but downweighted by source type. This is Reflective Memory Management
   (ACL 2025) and the Generative Agents reflection loop.
3. **Importance scoring at write time.** Generative Agents asks the LLM to
   rate each memory's importance 1–10 at creation. Currently we use
   classifier intensity as a proxy, which is correlated but not identical
   to "is this worth remembering?".
4. **Dynamic weight tuning per personality.** The default weights are reasonable
   but a "more topical" or "more mood-driven" lemon variant would just be
   different env values.

# References

* [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory (arXiv 2504.19413)](https://arxiv.org/abs/2504.19413)
* [State of AI Agent Memory 2026 (Mem0 blog)](https://mem0.ai/blog/state-of-ai-agent-memory-2026)
* [Generative Agents: Interactive Simulacra of Human Behavior (Park et al.)](https://dl.acm.org/doi/fullHtml/10.1145/3586183.3606763)
* [In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents (ACL 2025)](https://aclanthology.org/2025.acl-long.413/)
* [LoCoMo: Evaluating Very Long-Term Conversational Memory of LLM Agents (ICLR 2025)](https://arxiv.org/abs/2402.17753)
* [LongMemEval — long-term memory evaluation benchmark](https://www.emergentmind.com/topics/locomo-and-longmemeval-_s-benchmarks)
* [ClawMem — on-device hybrid memory layer (BM25 + vector + RRF)](https://github.com/yoloshii/ClawMem)
* [SQLite FTS5 documentation](https://www.sqlite.org/fts5.html)
* [Why SQLite+FTS5 beats vector DBs for AI agent memory](https://dev.to/fex_beck_27bfd4dccd05f062/why-sqlitefts5-beats-vector-dbs-for-ai-agent-memory-4inj)
* [Persode: episodic memory-aware journaling agent (2025)](https://arxiv.org/html/2508.20585v1)
* [Mood-Congruent Memory Revisited (PMC, 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10076454/)
* [Retrieval evaluation metrics: recall@K, MRR, NDCG (Pinecone)](https://www.pinecone.io/learn/offline-evaluation/)
* [RAG evaluation metrics 2025 (LangCopilot)](https://langcopilot.com/posts/2025-09-17-rag-evaluation-101-from-recall-k-to-answer-faithfulness)
