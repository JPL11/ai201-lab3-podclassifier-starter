"""
Optional challenge 2: tune the prompt systematically.

Pick the lowest-performing class and try three ways of PRESENTING the training
examples — without changing labels or the model — then measure per-class
accuracy for that class each time. This isolates how sensitive the few-shot
classifier is to example order and count.

Variants (the example LIST handed to classify_episode is the only thing that
changes; build_few_shot_prompt renders it in order):
  - baseline   : examples in natural (file) order
  - target_last: reorder so the target class's examples come LAST, nearest the
                 query (tests primacy/recency effects)
  - two_each   : only 2 examples per class (tests whether less, tighter context
                 helps or hurts the target class)

By default it evaluates only on the target class's test episodes to keep the
token cost down (e.g. 5 narrative episodes × 3 variants = 15 calls). Each full-
example variant costs ~2,860 tokens/call, so the full sweep is ~40k+ tokens —
run it on a fresh daily budget.

Usage:
    python exp_prompt_tuning.py --target narrative
    python exp_prompt_tuning.py --target narrative --variants baseline,target_last
"""

import argparse
import json
import os
from collections import defaultdict

from config import DATA_PATH, TEST_FILE, VALID_LABELS
from classifier import load_labeled_examples, classify_episode


def order_baseline(examples):
    return list(examples)


def order_target_last(examples, target):
    non_target = [e for e in examples if e["label"] != target]
    target_ex = [e for e in examples if e["label"] == target]
    return non_target + target_ex


def order_two_each(examples):
    by_class = defaultdict(list)
    for e in examples:
        by_class[e["label"]].append(e)
    out = []
    for label in VALID_LABELS:
        out.extend(by_class[label][:2])
    return out


VARIANTS = {
    "baseline": lambda ex, tgt: order_baseline(ex),
    "target_last": lambda ex, tgt: order_target_last(ex, tgt),
    "two_each": lambda ex, tgt: order_two_each(ex),
}


def evaluate_variant(examples, test_episodes, target):
    correct = 0
    for ep in test_episodes:
        pred = classify_episode(ep["description"], examples)
        ok = pred["label"] == ep["label"]
        correct += ok
        mark = "✅" if ok else f"❌(→{pred['label']})"
        print(f"    {mark} {ep['title'][:50]}")
    return correct, len(test_episodes)


def run(target: str, variant_names: list[str]):
    all_examples = load_labeled_examples()
    test_path = os.path.join(DATA_PATH, TEST_FILE)
    with open(test_path, encoding="utf-8") as f:
        test_episodes = json.load(f)
    target_episodes = [ep for ep in test_episodes if ep["label"] == target]

    print(f"\n=== Prompt tuning for '{target}' "
          f"({len(target_episodes)} test episodes) ===")
    summary = {}
    for name in variant_names:
        examples = VARIANTS[name](all_examples, target)
        print(f"\n-- variant: {name}  ({len(examples)} examples shown) --")
        correct, total = evaluate_variant(examples, target_episodes, target)
        summary[name] = (correct, total)
        print(f"   {target} accuracy: {correct}/{total} = {correct / total:.0%}")

    print("\n=== Summary ===")
    for name, (c, t) in summary.items():
        print(f"  {name:<14} {c}/{t}  ({c / t:.0%})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="narrative", choices=VALID_LABELS)
    p.add_argument("--variants", default="baseline,target_last,two_each")
    args = p.parse_args()
    run(args.target, args.variants.split(","))
