# Lab 3 — Optional Challenges

Each challenge has a runnable script. Results below are from real runs on
`llama-3.3-70b-versatile`, spread over two days to fit the Groq free tier
(~100k tokens/day; a single full 20-episode eval costs up to ~57k).

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

## Challenge 1 — Find the breaking point ✅ complete

`exp_breaking_point.py --n N` — shrink the training set to N examples/class
and re-evaluate.

| Config | Accuracy | Per-class |
|---|---|---|
| 5/class (baseline, full 20 test) | 100% | 5/5 all |
| 3/class (12-episode balanced subset) | 100% (12/12) | 3/3 all |
| 1/class (full 20 test) | 100% (20/20) | 5/5 all |
| **0/class — zero-shot** (full 20 test) | **95% (19/20)** | interview **4/5**, rest 5/5 |

**The breaking point is zero-shot, and `interview` breaks first** — not solo,
as we'd hypothesized from the confidence data. The one miss: *"Organizing at an
Amazon Warehouse: A First-Person Account"* was called solo — without examples
anchoring the host-guest structure, the model latched onto the first-person
surface framing. That's the same surface-vs-structure trap the adversarial
suite probes, and it confirms what makes the few-shot setup work: **a single
example per class is enough to fully recover it** (1/class → 100%).

---

## Challenge 3 — Confidence score ✅ run

`exp_confidence.py` — implemented the **token-free** route: derive a 0–10
confidence from hedging vs. decisive language in the reasoning the model already
returns (no extra API call, no prompt change). Aggregated per class on the
3/class run:

| Class | 3/class run | 1/class run | zero-shot run |
|---|---|---|---|
| narrative | 9.0 | 9.2 | 9.0 |
| interview | 8.7 | 7.6 | 7.8 |
| panel | 8.3 | 7.8 | 7.4 |
| **solo** | **7.0** | **6.4** | **6.8** |

**Solo is the least-confident class in every run** — it's defined negatively
("no guest, no external sources"), so the reasoning hedges more even when it's
right. And it *was* always right: solo held 100% even zero-shot.

The zero-shot run finally produced an error, so the correct-vs-wrong split
became measurable — and it's an honest **negative result**: the one *wrong*
prediction scored **9.0** confidence vs. **7.5** average for correct interview
predictions. Hedging-based confidence fails exactly on this error type — when
the model is fooled by surface framing, it isn't uncertain, it's confidently
wrong. Low confidence here signals a *hard class definition*, not a likely
error. A real deployment would need a second signal (e.g. self-rated certainty
or agreement across prompt variants) to catch confident misses.

---

## Challenge 2 — Tune the prompt systematically ✅ complete

`exp_prompt_tuning.py --target interview` — targeted at the class the
breaking-point sweep showed is weakest (interview, the only zero-shot miss;
the data overruled our earlier solo hypothesis). Three example-presentation
variants, evaluated on the 5 interview test episodes:

| Variant | What changes | interview accuracy |
|---|---|---|
| `baseline` | all 20 examples, natural order | 5/5 (100%) |
| `target_last` | interview examples moved last, nearest the query | 5/5 (100%) |
| `two_each` | only 2 examples/class (8 total) | 5/5 (100%) |

**The classifier is insensitive to example order and count** — recency
position and a 60% cut in examples changed nothing. Together with challenge 1,
the systematic conclusion: performance on this task is a step function in the
*presence* of examples (0 → 95%, ≥1/class → 100%), not a gradient in how
they're presented. Tuning effort should go into the taxonomy/edge-case rules
in the system prompt, not example curation.

---

## Summary

| # | Challenge | Status | Headline |
|---|---|---|---|
| 4 | Adversarial | ✅ complete | 4/4 traps resisted, incl. human-hard |
| 1 | Breaking point | ✅ complete | breaks only at zero-shot (95%); interview falls first; 1/class fully recovers |
| 3 | Confidence | ✅ complete | solo least confident in every run yet never wrong; the one error was *confidently* wrong — hedging is no error detector |
| 2 | Prompt tuning | ✅ complete | accuracy is a step function in example presence; order/count don't matter |
