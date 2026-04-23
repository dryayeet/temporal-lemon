# Benchmarking lemon

How to actually test lemon. Self-contained guide: read this when you want to measure quality, decide what to run, and stop wondering whether the pipeline is worth it.

---

## 1. What you are measuring

Two separate questions. Don't confuse them.

### 1.1 Ceiling: how good is the underlying model?

Raw Claude (Haiku / Sonnet / Opus), no lemon pipeline, no memory, no post-check. This is the honest baseline for MCQ-shaped benchmarks (EmoBench, ToMBench, Test of Time, TRAM, TimeBench, TimeQA) because the lemon stack is not what gets tested there, the model is.

**Why it matters:** if raw Claude Haiku already scores 85% on EQ-Bench and you swap it for Sonnet to get 92%, you learned that the floor moved. It tells you nothing about whether your pipeline helped.

### 1.2 Stack lift: does lemon's pipeline beat raw Claude?

Classifier + memory + ToM + post-check vs. the same model answering cold. Only benchmarks that fit are the multi-turn conversational ones (EQ-Bench 3, HEART). On MCQ benchmarks the pipeline has no room to do its job.

**Why it matters:** this is the interesting question. If the delta is small, the pipeline is not earning its 3 extra calls per turn. If the delta is large, you have evidence the stack is doing useful work.

---

## 2. Benchmark catalog

Quick reference. Pick from here based on what you want to answer.

### 2.1 Emotional intelligence / empathy

| Benchmark | Format | Good for | Verdict for lemon |
|---|---|---|---|
| **EQ-Bench 3** | 45 multi-turn roleplays, LLM-judged (Sonnet 3.7 judge) on 18 criteria | Stack lift AND ceiling | **Primary benchmark.** Stateful multi-turn matches lemon's design. Open-source leaderboard. |
| **HEART** | ESConv-derived multi-turn support chats, 5 dimensions | Stack lift for counseling angle | **Secondary benchmark.** Purpose-built for support-chatbot eval. |
| **EmoBench** (ACL 2024) | 400 hand-crafted MCQs, EU + EA tasks, EN + ZH | Ceiling only | Run on raw Claude. Lemon pipeline doesn't apply. |
| **EmoBench-M** (2025) | Multimodal extension | Ceiling only | Skip unless you add image input to lemon. |
| **ToMBench** | 2,860 Theory-of-Mind MCQs, 8 tasks / 31 abilities | Ceiling only | Run on raw Claude. MCQ. |
| **EPITOME** (Sharma et al., 2020) | 10k Reddit pairs, 3-dim empathy classifier | Training data, not a scored leaderboard | Use the classifier as a judge inside best-of-N, not as a benchmark run. |
| **EmpatheticDialogues / ESConv** | 25k / 1.3k conversations | Source corpora | Mine for test scenarios, don't treat as benchmark runs. |
| **CES-LCC** (2025) | 27-item rubric, 9 dimensions, eDelphi-validated | Qualitative counseling review | Best option for "is lemon clinically reasonable?" |
| **VERA-MH** (2025) | Clinician-designed safety rubric, suicide-risk focused | Safety evaluation | Run if you plan to positioning lemon near mental-health space. |

### 2.2 Temporal reasoning

| Benchmark | Format | Good for | Verdict for lemon |
|---|---|---|---|
| **Test of Time (ToT)** (Google, 2024) | Synthetic MCQ, no train-set leakage; two tracks: Semantic + Arithmetic | Ceiling only | Best-designed temporal benchmark. Raw Claude run. |
| **TRAM** (ACL Findings 2024) | 526.7k MCQs across 10 tasks (order, arithmetic, frequency, duration) | Ceiling only | Raw Claude. Volume is big, sample a subset. |
| **TimeBench** (2024) | 19k instances, 3 categories, 10 tasks | Ceiling only | Raw Claude. |
| **TempReason** | Fact-based vs. context-based probing | Ceiling only | Raw Claude. Useful to show where context-based reasoning degrades. |
| **TimeQA** | 14k+ Wikidata time-sensitive QA, Easy + Hard splits | Ceiling only | Raw Claude. |

**Note:** lemon has explicit temporal-reasoning machinery (`<time_context>` block, `<rules_of_time>` prompt section, `time_context.py`), so in principle the stack could be tested on temporal benchmarks. In practice the benchmarks are MCQ and lemon's pipeline is not shaped for MCQ answering. If you want to show the temporal scaffolding helps, build a custom temporal-dialogue benchmark instead.

### 2.3 Sources for each

- EQ-Bench: https://eqbench.com, https://github.com/EQ-bench/eqbench3, arXiv 2312.06281
- HEART: arXiv 2601.19922
- EmoBench: https://github.com/Sahandfer/EmoBench, ACL 2024
- ToMBench: https://github.com/zhchen18/ToMBench, arXiv 2402.15052
- EPITOME: https://github.com/behavioral-data/Empathy-Mental-Health, arXiv 2009.08441
- CES-LCC: MDPI Informatics 12/1/33 (2025)
- VERA-MH: arXiv 2510.15297
- Test of Time: https://research.google/pubs/test-of-time-benchmarking-llms-on-temporal-reasoning/, arXiv 2406.09170
- TRAM: ACL Findings 2024, https://aclanthology.org/2024.findings-acl.382.pdf
- TimeBench: arXiv 2311.17667

---

## 3. Practical order

Do these in sequence. Each step answers something the next step assumes.

1. **Raw Claude on EQ-Bench 3 + Test of Time.** Establishes your ceiling in an afternoon. Should cost ~$5-15 depending on model.
2. **Lemon stack on EQ-Bench 3, pipeline ON vs OFF.** Answers: does the pipeline actually help? This is the most honest test you can run.
3. **Lemon on HEART.** Tests the counseling-support angle specifically. More clinical than EQ-Bench.
4. **CES-LCC rubric on ~20 hand-picked scenarios.** Qualitative clinician-style review. Catches things automated benchmarks miss.

Skip TRAM / TimeBench / TempReason / EmoBench / ToMBench for lemon itself. They are MCQ, your stack is not the variable being tested there. Run them on raw Claude if you want the model-level numbers for `docs/empathy_research.md`.

---

## 4. Ceiling test recipe

Run the model that lemon uses, cold, on whichever benchmark you chose.

### 4.1 Find your current model ID

```bash
grep CHAT_MODEL src/config.py
# LEMON_CHAT_MODEL default: anthropic/claude-haiku-4.5
```

That's what lemon ships with. Run the benchmark against the same ID (via OpenRouter with your existing `OPENROUTER_API_KEY`, or directly via Anthropic with an `ANTHROPIC_API_KEY`).

### 4.2 EQ-Bench 3 ceiling run

```bash
# somewhere outside the lemon repo
git clone https://github.com/EQ-bench/eqbench3
cd eqbench3
pip install -r requirements.txt

# follow the repo's README to configure the model client. As of writing it
# supports OpenAI-compatible endpoints — OpenRouter is drop-in.
export OPENAI_API_KEY=$OPENROUTER_API_KEY   # if using OpenRouter
export OPENAI_API_BASE=https://openrouter.ai/api/v1

python run_eqbench.py \
    --model "anthropic/claude-haiku-4.5" \
    --output results/haiku-4.5.json
```

Upload results to the public leaderboard if you want, or just keep the score. Note the 18 per-dimension sub-scores, not just the aggregate.

### 4.3 Test of Time ceiling run

```bash
# datasets available on HF: "baharef/ToT"
git clone https://github.com/DKMS-SpaceBiology/test-of-time
# or pull the dataset directly and roll your own harness
```

Run separately on `ToT-Semantic` and `ToT-Arithmetic`. Record accuracy for each.

### 4.4 What to save

For every ceiling run, save at minimum:
- Model ID
- Date of run
- Full per-question output log
- Aggregate score + sub-scores

Put these in a git-ignored `evaluation/` directory so you have a time-series of how your ceiling moves when you swap models.

---

## 5. Stack-lift test recipe

This is the interesting one. Test whether lemon's pipeline adds value over raw Claude.

### 5.1 Build a benchmark adapter

A ~50-line script that takes a benchmark's user-turn input and returns lemon's reply. Drop it at `src/bench_adapter.py`:

```python
"""Benchmark adapter: run a single user turn through the lemon pipeline in
isolation. Used by external benchmark harnesses (EQ-Bench, HEART) that expect
a text-in / text-out model client.

Each invocation starts a FRESH session so benchmark scenarios don't contaminate
each other.
"""
from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import config
import db
from commands import ChatContext
from facts import format_user_facts
from history import replace_system_block
from pipeline import run_empathy_turn
from prompt import LEMON_PROMPT
from state import DEFAULT_STATE, format_internal_state
from time_context import get_time_context


def one_turn(user_msg: str, prior_turns: list[dict] | None = None) -> str:
    """Run one user turn through the pipeline in a fresh isolated session.

    `prior_turns`, if given, is the benchmark's simulated conversation history
    as [{"role":"user","content":...}, {"role":"assistant","content":...}, ...].
    These are prepended to the lemon history AS-IF they happened in this session.
    """
    # isolate DB to a throwaway file so this run does not bleed into .lemon.db
    with tempfile.TemporaryDirectory() as tmp:
        config.DB_PATH = Path(tmp) / "bench.db"

        session_start = datetime.now()
        sid = db.start_session()
        state = dict(DEFAULT_STATE)

        history: list[dict] = [
            {"role": "system", "content": LEMON_PROMPT},
            {"role": "system", "content": get_time_context(session_start)},
            {"role": "system", "content": format_internal_state(state)},
        ]
        if prior_turns:
            history.extend(prior_turns)
            for m in prior_turns:
                db.log_message(sid, m["role"], m["content"])

        reply, _trace = run_empathy_turn(
            user_msg=user_msg,
            base_history=history,
            energy=state["energy"],
            model=config.CHAT_MODEL,
            session_id=sid,
            keep_recent_turns=config.KEEP_RECENT_TURNS,
            on_phase=None,
        )
        return reply


if __name__ == "__main__":
    # smoke test
    print(one_turn("i had a really rough day at work"))
```

**Why one-shot isolation:** benchmark scenarios are independent. If you reuse the same session, the memory-retrieval step can pull emotion-tagged messages from scenario 7 while scoring scenario 12, which silently contaminates results.

### 5.2 Wire the adapter into EQ-Bench 3

Fork `eqbench3`, find its model-client abstraction (typically `model_client.py` or `api.py`), replace the completion call with a Python invocation of your adapter:

```python
# in the eqbench3 fork, client layer
import subprocess, json

def complete(messages):
    # messages is [{"role":..., "content":...}, ...] ending with the user's turn
    user_msg = messages[-1]["content"]
    prior_turns = messages[:-1]
    # call the adapter in a subprocess so the benchmark harness stays unaware
    # of lemon's internal imports
    out = subprocess.check_output([
        "python", "-c",
        f"import sys; sys.path.insert(0,'/path/to/lemon/src'); "
        f"from bench_adapter import one_turn; "
        f"import json; "
        f"print(one_turn({json.dumps(user_msg)}, {json.dumps(prior_turns)}))",
    ], text=True)
    return out.strip()
```

(If the benchmark's client layer takes a Python callable, just import `one_turn` directly.)

### 5.3 Run pipeline ON vs pipeline OFF

Two runs on the same benchmark, same scenarios, same model. Only variable: `LEMON_EMPATHY`.

```bash
# Pipeline ON (classifier + memory + ToM + post-check)
LEMON_EMPATHY=1 python run_eqbench.py --model lemon-adapter --output results/lemon-on.json

# Pipeline OFF (thin wrapper around chat.generate_reply)
LEMON_EMPATHY=0 python run_eqbench.py --model lemon-adapter --output results/lemon-off.json
```

The delta between the two runs is the answer to "does the pipeline help?".

### 5.4 Interpretation

Look at the per-dimension sub-scores, not just the aggregate.

- **If empathy/insight/social-acuity go up but the aggregate is flat:** the pipeline is doing its job but cost came out of other dimensions (fluency, task-following). Fine trade.
- **If the aggregate goes up:** pipeline is a strict improvement. Keep it on.
- **If the aggregate goes down:** the post-check regex is firing false positives, or the ToM scaffolding is constraining the model too much. Inspect failed cases.
- **If all scores move by less than noise (~1-2 points):** the pipeline is not earning its latency. Consider turning it off by default.

### 5.5 Gotchas

- **Prompt caching.** If you run the same benchmark twice in quick succession against an Anthropic model, the second run hits the cache. This is expected behavior and does not bias scores, but it makes per-run cost look weirdly low the second time.
- **Memory contamination.** Confirm every scenario runs with a fresh `config.DB_PATH` (the adapter above does this via `tempfile.TemporaryDirectory`).
- **Humanize pacing.** Disable it for benchmark runs. The typing delays are pure latency with no effect on content. Set `LEMON_HUMANIZE=0`.
- **State updater.** The state updater runs every 2 turns. In a one-shot benchmark it fires once and produces an irrelevant snapshot. Consider setting `STATE_UPDATE_EVERY=999` inside the adapter to skip it entirely.
- **Phase callback.** `on_phase=None` in the adapter. Benchmarks don't care.

---

## 6. HEART recipe

HEART is purpose-built for support-chatbot evaluation, specifically multi-turn. Evaluation is along five dimensions: human alignment, empathic responsiveness, attunement, resonance, task-following.

HEART uses emotionally-complex scenarios drawn from ESConv (grief, frustration, conflict, uncertainty). It is newer and the tooling is less mature than EQ-Bench, but if you care about the counseling framing specifically, it is closer to the ground truth than EQ-Bench's roleplay scenarios.

Process:

1. Pull the paper (arXiv 2601.19922) and the benchmark scripts from its repo.
2. Wire the same `bench_adapter.one_turn` as above.
3. Run pipeline ON vs OFF.
4. Attunement and resonance are the dimensions where lemon's memory + ToM scaffolding should show the largest lift. If they do not, the scaffolding is probably not rich enough.

---

## 7. Custom rubric: CES-LCC

If you are positioning lemon as a counseling chatbot, CES-LCC is the framework clinicians will actually grade you against. 27 items across 9 dimensions: Understanding Requests, Providing Helpful Information, Clarity and Relevance of Responses, Language Quality, Trust, Emotional Support, Guidance and Direction, Memory, and Overall Satisfaction.

### 7.1 Setup

Hand-pick 15-20 scenarios representative of the kinds of conversations lemon should handle well (and some it should handle gracefully even if not "well"):

- Flat small-talk
- Emotionally loaded disclosure
- Advice-seeking
- Ambiguous / confusing messages
- Repeated returns to the same topic across turns (tests memory)
- Edge cases: very short messages, very long messages, non-English
- Mild safety-adjacent: user says "I've been really low lately"

Script each as a mini-conversation (3-6 turns), end on a user turn.

### 7.2 Run lemon

Use `bench_adapter.one_turn` with `prior_turns` set to the earlier turns of each scenario. Capture the final reply.

### 7.3 Judge

Write an LLM judge prompt with the 27 CES-LCC items rendered as Yes/No/Partial questions. Example:

```
You are scoring a chatbot reply against the CES-LCC rubric.

Scenario:
[scenario context + conversation so far]

Chatbot's reply:
[lemon's reply]

Rate each item Yes / Partial / No:

UNDERSTANDING REQUESTS
1. Did the reply correctly identify what the user was asking for?
2. Did the reply avoid misinterpreting the user's intent?
...

EMOTIONAL SUPPORT
10. Did the reply acknowledge the user's feelings?
11. Did the reply avoid dismissing or minimizing the feelings?
12. Was the emotional register appropriate (not too cheerful, not too clinical)?
...

MEMORY
22. Did the reply reference earlier turns when relevant?
23. Did the reply avoid contradicting earlier statements?
...

Return JSON: {"item_1": "Yes", "item_2": "Partial", ...}
```

Use Claude Opus or Sonnet as the judge. Score aggregates per dimension.

### 7.4 Iterate

The CES-LCC pattern where lemon consistently loses points reveals what to fix. Common losers for stack chatbots:

- Memory (item 22-23) if the memory retrieval threshold is too strict.
- Language Quality if the model is hitting weird tone overrides from the persona block.
- Guidance and Direction if the `<conversation rules>` "no advice unless asked" rule is firing too aggressively.

---

## 8. VERA-MH (safety)

Run this if lemon is exposed to users outside your control. VERA-MH is specifically about suicide-risk handling. The judge-agent categorizes each reply as best-practice / missed-opportunity / actively-damaging / not-relevant.

Process mirrors CES-LCC: a set of prompts (this time focused on distress), lemon replies, LLM judge scores. The interesting metric is the **actively-damaging rate**. Anything above zero is a flag.

Do not rely on this alone for a clinical deployment. It is an automated sanity check, not a clinical validation.

---

## 9. What to write down

For every run, keep:

- **Run metadata:** date, git commit, model ID, pipeline on/off, `LEMON_*` env vars
- **Raw outputs:** full per-scenario inputs and outputs
- **Aggregate scores + per-dimension scores**
- **Failure cases:** lowest-scoring scenarios with human notes on why

Suggested layout:

```
evaluation/
  2026-04-23_eqbench3_haiku_pipeline-on.json
  2026-04-23_eqbench3_haiku_pipeline-off.json
  2026-04-23_eqbench3_haiku_ceiling.json
  2026-04-23_ceslcc_haiku_pipeline-on.md
  README.md     # index, what each file is, which commit
```

Add `evaluation/` to `.gitignore` or commit the summaries only (not per-question transcripts, which can get large and contain generated content you might not want in git history).

---

## 10. Common pitfalls

- **Testing raw Claude and calling it "lemon." ** Raw-Claude numbers are lemon's ceiling, not lemon's score. Label every run unambiguously.
- **Running the benchmark twice and averaging.** LLMs are non-deterministic even at `temperature=0`. If you want confidence intervals, run N=5 with different seeds and report mean + std.
- **Mixing pipeline-on and pipeline-off in the same run.** The `LEMON_EMPATHY` flag is read from `config.py` at import time; once imported, flipping the env var does nothing. Restart the process per run.
- **Using a more capable model as the judge than the generator.** This is correct for quality scoring. But a Haiku-judged Haiku run gives a very different number than a Sonnet-judged Haiku run. Document the judge.
- **Letting the scenario corpus leak between runs.** If you tune lemon against scenarios from `evaluation/scenarios-v1.yaml`, do the final evaluation against a held-out `scenarios-v2.yaml`. Otherwise you are optimizing on your test set.
- **Trusting aggregate scores on MCQ benchmarks for a chat product.** MCQ accuracy on EmoBench tells you almost nothing about conversational quality. Keep those numbers in a separate column.

---

## 11. TL;DR decision tree

```
Want a single number to report?
  → EQ-Bench 3 with pipeline ON.

Want to know if the pipeline is worth its latency?
  → EQ-Bench 3 pipeline ON vs OFF on the same model.

Want the model-level ceiling?
  → raw Claude on EQ-Bench 3 (ignore pipeline entirely).

Want temporal-reasoning ceiling?
  → raw Claude on Test of Time.

Want clinical-style review?
  → CES-LCC rubric, 20 hand-picked scenarios, LLM judge.

Want safety signal?
  → VERA-MH.

Want to know if lemon beats a specific competitor product?
  → roll your own head-to-head with an LLM preference judge.
  No existing benchmark measures "friend-likeness."
```

---

## 12. Quick reference

**Run ceiling on EQ-Bench 3:**

```bash
git clone https://github.com/EQ-bench/eqbench3 && cd eqbench3
export OPENAI_API_KEY=$OPENROUTER_API_KEY
export OPENAI_API_BASE=https://openrouter.ai/api/v1
python run_eqbench.py --model anthropic/claude-haiku-4.5 --output results/ceiling.json
```

**Build the adapter:**

```bash
# create src/bench_adapter.py (see §5.1)
LEMON_HUMANIZE=0 python -c "from bench_adapter import one_turn; print(one_turn('hey'))"
```

**Pipeline ON vs OFF:**

```bash
LEMON_EMPATHY=1 LEMON_HUMANIZE=0 python run_eqbench.py --model lemon-adapter --output results/on.json
LEMON_EMPATHY=0 LEMON_HUMANIZE=0 python run_eqbench.py --model lemon-adapter --output results/off.json
diff <(jq .aggregate results/on.json) <(jq .aggregate results/off.json)
```

**Score a CES-LCC scenario:**

```bash
# assemble rubric prompt (see §7.3)
# call your preferred judge API
# save JSON per scenario, aggregate manually or via a small script
```

**Tag the git commit for every run:**

```bash
git rev-parse --short HEAD  # stamp this into the results file metadata
```
