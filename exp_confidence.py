"""
Optional challenge 3: confidence scoring.

The challenge offers two routes: ask the LLM to rate its own certainty (costs
extra tokens / a prompt change), or infer confidence from hedging language in
the reasoning it already produced. We take the second, token-free route — it
reuses the reasoning string every classification already returns, so it adds
zero API cost (which matters on the free tier's 100k tokens/day).

confidence_from_reasoning() returns a 0–10 score. per_class_confidence()
aggregates it the same way compute_per_class_accuracy() aggregates correctness,
so it can sit alongside the accuracy report.
"""

import re

# Words/phrases that signal the model is unsure — each one lowers confidence.
_HEDGES = [
    "might", "may ", "maybe", "could", "possibly", "perhaps", "unclear",
    "ambiguous", "not sure", "hard to", "difficult to", "seems", "appears",
    "arguably", "somewhat", "borderline", "uncertain", "tends to", "leaning",
    "either", "on the other hand", "however", "although", "though", "unsure",
    "or it could", "or possibly", "between", "blurs", "edge case",
]
# Phrases that signal a confident, structural read — each one raises confidence.
_STRONG = [
    "clearly", "clear ", "definitely", "explicitly", "unambiguous", "directly",
    "host-guest", "single guest", "one guest", "no guest", "multiple guests",
    "roughly equal", "story arc", "assembled from", "the host speaks with",
    "indicates", "strongly", "is structured as",
]


def confidence_from_reasoning(reasoning: str) -> int:
    """Heuristic 0–10 confidence from hedging vs. decisive language."""
    if not reasoning:
        return 0
    text = reasoning.lower()
    hedges = sum(text.count(h) for h in _HEDGES)
    strong = sum(text.count(s) for s in _STRONG)
    score = 7 - 2 * hedges + strong
    return max(0, min(10, score))


def per_class_confidence(results: list[dict]) -> dict[str, dict]:
    """
    Average confidence grouped by ground-truth label, plus a split of average
    confidence on correct vs. incorrect predictions (to test the hypothesis
    that low confidence correlates with misclassification).
    """
    from config import VALID_LABELS

    stats = {label: {"scores": [], "correct": [], "wrong": []} for label in VALID_LABELS}
    for r in results:
        truth = r["ground_truth"]
        if truth not in stats:
            continue
        c = confidence_from_reasoning(r.get("reasoning", ""))
        stats[truth]["scores"].append(c)
        (stats[truth]["correct"] if r.get("correct") else stats[truth]["wrong"]).append(c)

    out = {}
    for label, s in stats.items():
        avg = sum(s["scores"]) / len(s["scores"]) if s["scores"] else 0.0
        out[label] = {
            "avg_confidence": round(avg, 1),
            "n": len(s["scores"]),
            "avg_when_correct": round(sum(s["correct"]) / len(s["correct"]), 1) if s["correct"] else None,
            "avg_when_wrong": round(sum(s["wrong"]) / len(s["wrong"]), 1) if s["wrong"] else None,
        }
    return out


def format_confidence_report(results: list[dict]) -> str:
    pc = per_class_confidence(results)
    lines = ["**Average confidence per class (0–10, from reasoning hedging):**"]
    for label, s in pc.items():
        bar = "▮" * int(s["avg_confidence"]) + "▯" * (10 - int(s["avg_confidence"]))
        extra = ""
        if s["avg_when_wrong"] is not None:
            extra = f"  (correct: {s['avg_when_correct']}, wrong: {s['avg_when_wrong']})"
        lines.append(f"  {label:<12} {bar}  {s['avg_confidence']}/10  (n={s['n']}){extra}")
    return "\n".join(lines)
