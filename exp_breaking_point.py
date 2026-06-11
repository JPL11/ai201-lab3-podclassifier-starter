"""
Optional challenge 1: find the breaking point.

Re-run evaluation with a SMALLER training set (N labeled examples per class
instead of the full 5) and see how accuracy degrades and which class falls
apart first. Also reports per-class confidence (challenge 3) from the same run,
so one eval covers both challenges.

Token-aware: each call's cost scales with N (fewer examples = cheaper), and you
can cap the test set with --test-limit to fit a tight daily budget. API
failures (e.g. rate-limit 429s) are reported as errors and excluded from
accuracy rather than counted as wrong.

Usage:
    python exp_breaking_point.py --n 3            # 3 examples/class, full test
    python exp_breaking_point.py --n 3 --test-limit 12
"""

import argparse
import json
import os
from collections import defaultdict

from config import DATA_PATH, TEST_FILE, VALID_LABELS
from classifier import load_labeled_examples, classify_episode
from evaluate import compute_accuracy, compute_per_class_accuracy
from exp_confidence import format_confidence_report


def subset_examples(examples: list[dict], n_per_class: int) -> list[dict]:
    """Keep the first n_per_class examples of each label (deterministic)."""
    by_class = defaultdict(list)
    for ex in examples:
        by_class[ex["label"]].append(ex)
    kept = []
    for label in VALID_LABELS:
        kept.extend(by_class[label][:n_per_class])
    return kept


def run(n_per_class: int, test_limit: int | None = None) -> dict:
    all_examples = load_labeled_examples()
    examples = subset_examples(all_examples, n_per_class)

    test_path = os.path.join(DATA_PATH, TEST_FILE)
    with open(test_path, encoding="utf-8") as f:
        test_episodes = json.load(f)
    if test_limit:
        # Balanced subset: first test_limit//4 of each class, by reading order.
        per = max(1, test_limit // len(VALID_LABELS))
        seen = defaultdict(int)
        picked = []
        for ep in test_episodes:
            if seen[ep["label"]] < per:
                picked.append(ep)
                seen[ep["label"]] += 1
        test_episodes = picked

    print(f"\n=== Breaking point: {n_per_class} example(s)/class "
          f"({len(examples)} total), {len(test_episodes)} test episodes ===")

    results = []
    for ep in test_episodes:
        pred = classify_episode(ep["description"], examples)
        api_error = pred["label"] == "unknown" and pred["reasoning"].startswith("Classification error:")
        results.append({
            "ground_truth": ep["label"],
            "predicted": pred["label"],
            "reasoning": pred["reasoning"],
            "correct": (not api_error) and pred["label"] == ep["label"],
            "api_error": api_error,
        })
        mark = "⚠️ " if api_error else ("✅" if results[-1]["correct"] else "❌")
        print(f"  {mark} {ep['label']:<10} → {pred['label']:<10} {ep['title'][:46]}")

    errors = sum(r["api_error"] for r in results)
    scored = [r for r in results if not r["api_error"]]
    preds = [r["predicted"] for r in scored]
    truth = [r["ground_truth"] for r in scored]

    acc = compute_accuracy(preds, truth)
    per_class = compute_per_class_accuracy(preds, truth)

    print(f"\nOverall accuracy: {acc:.0%} ({sum(r['correct'] for r in scored)}/{len(scored)})"
          + (f"   ⚠️ {errors} API error(s) excluded" if errors else ""))
    for label, s in per_class.items():
        bar = "█" * int(s["accuracy"] * 10) + "░" * (10 - int(s["accuracy"] * 10))
        print(f"  {label:<12} {bar}  {s['accuracy']:.0%}  ({s['correct']}/{s['total']})")
    print()
    print(format_confidence_report(scored))

    return {"n_per_class": n_per_class, "accuracy": acc, "per_class": per_class,
            "errors": errors, "results": results}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3, help="examples per class")
    p.add_argument("--test-limit", type=int, default=None, help="cap # test episodes")
    args = p.parse_args()
    run(args.n, args.test_limit)
