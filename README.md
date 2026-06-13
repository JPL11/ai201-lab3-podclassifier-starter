# 🎙️ Pod Classifier

A few-shot podcast episode classifier. Given an episode description, it classifies the episode's format as `interview`, `solo`, `panel`, or `narrative` using labeled examples and an LLM (Groq `llama-3.3-70b-versatile`) — no fine-tuning, just a carefully built few-shot prompt.

**Status: complete.** All three milestones are done — **100% accuracy (20/20) on the held-out test set** — and all four optional challenges are implemented and run. Built for AI201 Lab 3.

---

## How It Works

```
data/my_labels.json ─→ load_labeled_examples() ─┐
                                                ├─→ build_few_shot_prompt() ─→ Groq LLM ─→ {label, reasoning}
episode description ────────────────────────────┘
```

| Milestone | What was done |
|---|---|
| **1 — Labeling** | Hand-labeled 20 training episodes against `data/taxonomy.md`, which defines the four formats *structurally* (who speaks, and where the material comes from) plus explicit edge-case rules — e.g. a self-interview is `solo`, a story assembled from external sources is `narrative` even in first person. |
| **2 — Classifier** | `classifier.py` builds a few-shot prompt from the labeled examples plus the taxonomy's edge-case rules and asks for a label *with reasoning*. Output is parsed and validated against the four labels; API failures degrade to an explicit error result instead of a fake label. |
| **3 — Evaluation** | `evaluate.py` computes overall and per-class accuracy over the 20-episode held-out test set, streamed live in the Gradio UI. **Result: 100% overall, 5/5 in every class.** |

### Results at a glance

| Question | Answer (measured) |
|---|---|
| Baseline accuracy | **100%** (20/20), all four classes 5/5 |
| Where does it break? | Only at **zero-shot**: 95%, and `interview` falls first |
| How many examples are needed? | **One per class** fully restores 100% |
| Does example order/count matter? | No — accuracy is a step function in example *presence* |
| Can it be fooled by surface wording? | No — 4/4 adversarial traps resisted |

---

## Optional Challenges

All four are implemented as standalone scripts, run for real, and written up in [`optional-challenges.md`](optional-challenges.md):

| Script | Challenge | Headline finding |
|---|---|---|
| `exp_breaking_point.py` | Shrink the training set until accuracy breaks | Still 100% at 3/class and **1/class**; zero-shot drops to 95%. The one miss: a first-person interview misread as `solo` — exactly the surface-vs-structure trap the examples exist to prevent. |
| `exp_prompt_tuning.py` | Tune example presentation systematically | Baseline order, target-class-last, and 2-per-class all score 5/5 on the weakest class. Tuning effort belongs in the taxonomy rules, not example curation. |
| `exp_confidence.py` | Score classifier confidence | Token-free approach: confidence derived from hedging language in the model's own reasoning. `solo` is least confident in every run yet never wrong — and the one real error was *more* confident than the correct answers. Honest negative result: hedging is not an error detector. |
| `exp_adversarial.py` | Descriptions written to fool the classifier | **4/4 resisted**, including a human-hard case. The reasoning cited the structural signal every time — the classifier reads structure, not keywords. |

Reproduce with:

```bash
python exp_breaking_point.py --n 1      # or --n 0 for zero-shot, --test-limit 12 to cap cost
python exp_prompt_tuning.py --target interview
python exp_adversarial.py
```

> **Token budget note:** the Groq free tier allows ~100k tokens/day and a full 20-episode eval costs up to ~57k — the experiment scripts take `--test-limit` / `--n` flags to fit runs into the budget.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY (free at console.groq.com) to .env
```

## Run

```bash
python app.py
```

Opens the Gradio UI: classify a single description interactively, or run the full evaluation with live streaming results.

---

## Project Structure

```
podclassifier/
├── app.py                   # Gradio UI: single classify + live eval
├── classifier.py            # Few-shot prompt construction + Groq classification
├── evaluate.py              # Overall + per-class accuracy
├── config.py                # Model, paths, labels, file names
├── exp_breaking_point.py    # Challenge 1: training-set size sweep
├── exp_prompt_tuning.py     # Challenge 2: example-presentation variants
├── exp_confidence.py        # Challenge 3: token-free confidence scoring
├── exp_adversarial.py       # Challenge 4: surface-vs-structure traps
├── optional-challenges.md   # Full experimental writeup
├── data/
│   ├── taxonomy.md          # Label definitions + edge-case rules (the real workhorse)
│   ├── train_episodes.json  # 20 training episodes
│   ├── my_labels.json       # Hand labels (Milestone 1)
│   └── test_episodes.json   # 20 held-out, pre-labeled test episodes
└── specs/
    ├── system-design.md
    ├── classifier-spec.md
    └── evaluation-spec.md
```
