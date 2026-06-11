"""
Optional challenge 4: adversarial descriptions.

Each case is written to signal one format on the surface while structurally
belonging to another (per data/taxonomy.md). `expected` is the structurally
correct label; `surface` is the trap the wording sets. The last case is a
"human-hard" one — subtle enough to fool a careful annotator, not just the model.

Usage: python exp_adversarial.py
"""

from classifier import load_labeled_examples, classify_episode

CASES = [
    {
        "title": "Interview: The Voice in My Head",
        "surface": "interview",
        "expected": "solo",
        "why": "Title says 'Interview' and uses Q&A framing, but there is no "
               "guest — the host questions and answers themselves.",
        "description": (
            "This week, no guest — just me, interviewing myself about the year "
            "I almost walked away from everything. I pose the hard questions and "
            "I answer them, alone at the mic, the way I wish someone had asked me "
            "at the time."
        ),
    },
    {
        "title": "The Archive of Her",
        "surface": "solo",
        "expected": "narrative",
        "why": "First-person and deeply personal (reads as solo), but the story "
               "is assembled from EXTERNAL sources — diaries, letters, and "
               "interviews with other people.",
        "description": (
            "I tell the story of my mother's secret life in the first person, the "
            "way only a daughter could. But every scene here was reconstructed "
            "from her diaries, the letters she never sent, and the recorded "
            "interviews I conducted with her old colleagues and her sister."
        ),
    },
    {
        "title": "Five Founders, One Stage",
        "surface": "panel",
        "expected": "interview",
        "why": "'Five founders' and 'panel' scream panel, but the episode is "
               "really one subject (Mara) being drawn out by the host while the "
               "others mostly listen — a host-guest dynamic.",
        "description": (
            "We gathered five startup founders for what looked like a panel. But "
            "this episode is really about one of them — Mara — as the host walks "
            "her through the slow collapse of her company, question by question. "
            "The other four are in the room, but they mostly listen."
        ),
    },
    {
        "title": "After the Whistle",
        "surface": "interview / panel (human-hard)",
        "expected": "narrative",
        "why": "'We hear their accounts in their own words' sounds like an "
               "interview/panel, but it's a reported story assembled from those "
               "accounts PLUS memos and statements, unfolding across a decade.",
        "description": (
            "A reported look at what happened to the three workers who blew the "
            "whistle on the plant. We hear their accounts in their own words, "
            "stitched together with internal memos and the company's public "
            "statements, as the timeline unfolds across a decade."
        ),
    },
]


def run():
    examples = load_labeled_examples()
    correct = 0
    for c in CASES:
        pred = classify_episode(c["description"], examples)
        hit = pred["label"] == c["expected"]
        correct += hit
        print(f"\n=== {c['title']}")
        print(f"  surface signal : {c['surface']}")
        print(f"  expected       : {c['expected']}")
        print(f"  predicted      : {pred['label']}  {'✅ not fooled' if hit else '❌ FOOLED'}")
        print(f"  model reasoning: {pred['reasoning'][:200]}")
        print(f"  (why adversarial: {c['why']})")
    print(f"\nClassifier resisted the trap on {correct}/{len(CASES)} cases.")


if __name__ == "__main__":
    run()
