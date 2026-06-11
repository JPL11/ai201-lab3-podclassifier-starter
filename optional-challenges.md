# Lab 3 — Optional Challenges

Each challenge has a runnable script. Results below are from real runs on
`llama-3.3-70b-versatile` unless marked *deferred* (the Groq free tier allows
~100k tokens/day, and a single full 20-episode eval already costs ~57k, so the
heavier sweeps are budget-bound to one-per-day).

Baseline (Milestone 3): **100% accuracy (20/20)**, all four classes 5/5.

---

## Challenge 4 — Adversarial descriptions ✅ run

`exp_adversarial.py` — four descriptions that signal one format on the surface
but structurally belong to another (per `data/taxonomy.md`).

| Case | Surface signal | Correct label | Predicted | Result |
|---|---|---|---|---|
| "Interview: The Voice in My Head" (self-interview, no guest) | interview | solo | solo | ✅ not fooled |
| "The Archive of Her" (first-person, but built from diaries/letters/others' interviews) | solo | narrative | narrative | ✅ not fooled |
| "Five Founders, One Stage" (5 founders, but one is the subject) | panel | interview | interview | ✅ not fooled |
| "After the Whistle" (*human-hard*: reported story w/ workers' accounts) | interview/panel | narrative | narrative | ✅ not fooled |

**Result: 4/4 — the classifier resisted every trap**, including the human-hard
case. Its reasoning explicitly cited the structural signal each time ("no
external guests or sources", "assembled from external sources", "host-guest
dynamic… the other four mostly listen", "stitched together with internal
memos"). Takeaway: the taxonomy's edge-case rules, carried into the prompt,
make the model classify by **structure**, not surface keywords — exactly what
keyword-matching would get wrong.

---

## Challenge 1 — Find the breaking point ✅ run (partial), deeper sweep deferred

`exp_breaking_point.py --n 3 --test-limit 12` — drop training from 5 to **3
examples/class** and re-evaluate (12-episode balanced test subset, to fit the
token budget).

| Config | Accuracy | Per-class |
|---|---|---|
| 5/class (baseline, full 20 test) | 100% | 5/5 all |
| **3/class** (12 test) | **100% (12/12)** | 3/3 all |

**It didn't break at 3/class.** That's itself the finding: with a clean
taxonomy and prototypical examples, 3 demonstrations per class already saturate
this test set. The true breaking point is below 3 — the natural next step is
`--n 1` (and `--n 2`), which is *deferred* only because today's token budget is
spent. Hypothesis for which class degrades first: **solo**, since it's the
residual "one voice, no external sources" category and is the most likely to be
confused with narrative when examples thin out (see its lower confidence below).

---

## Challenge 3 — Confidence score ✅ run

`exp_confidence.py` — implemented the **token-free** route: derive a 0–10
confidence from hedging vs. decisive language in the reasoning the model already
returns (no extra API call, no prompt change). Aggregated per class on the
3/class run:

| Class | Avg confidence |
|---|---|
| narrative | 9.0 / 10 |
| interview | 8.7 / 10 |
| panel | 8.3 / 10 |
| **solo** | **7.0 / 10** |

**Solo is the least-confident class**, consistent with the breaking-point
hypothesis: it's defined negatively ("no guest, no external sources"), so the
model's reasoning hedges more even when it's right. All predictions here were
correct, so we can't yet measure the low-confidence→misclassification
correlation — that needs a run with errors in it (e.g. the deferred `--n 1`
breaking-point), where `per_class_confidence()` already splits avg confidence on
correct vs. wrong predictions to test exactly that.

---

## Challenge 2 — Tune the prompt systematically ⏳ deferred (implemented)

`exp_prompt_tuning.py` — harness ready, but **not run today**: it needs ~15 LLM
calls (3 example-presentation variants × the 5 narrative test episodes) ≈ 40k+
tokens, which exceeds the remaining daily budget. Running it now would just hit
the rate limit.

Two reasons it's also the *right* one to defer:
1. The sweep targets the **lowest-performing class** to improve it — but every
   class is currently at 100%, so there's no degradation to tune away. It only
   becomes meaningful on a config that actually breaks (the deferred `--n 1`).
2. The three variants it tries — `baseline`, `target_last` (target class's
   examples nearest the query), and `two_each` (fewer, tighter examples) — are
   designed to measure order/recency and count sensitivity, which only show up
   once accuracy has headroom to move.

**To run (fresh daily budget):**
```bash
python exp_breaking_point.py --n 1                          # find a class that breaks
python exp_prompt_tuning.py --target solo                   # then tune the weak one
```

---

## Summary

| # | Challenge | Status | Headline |
|---|---|---|---|
| 4 | Adversarial | ✅ run | 4/4 traps resisted, incl. human-hard |
| 1 | Breaking point | ✅ run (3/class) | still 100% at 3/class; <3 deferred |
| 3 | Confidence | ✅ run | solo least confident (7.0), narrative most (9.0) |
| 2 | Prompt tuning | ⏳ implemented, deferred | needs fresh token budget; no weak class to tune yet |
