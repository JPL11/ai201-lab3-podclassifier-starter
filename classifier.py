import json
import os
import re
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.

    TODO — Milestone 2:

    Your prompt needs to:
      1. Describe the task and the four valid labels
      2. Show the labeled training examples so the LLM can learn the pattern
      3. Present the new description and ask for a classification

    The LLM should return a single label from VALID_LABELS (exactly as written)
    plus a brief explanation of its reasoning. Think carefully about the output
    format you request — you'll need to parse it in classify_episode().

    Before writing code, complete specs/classifier-spec.md.
    """
    lines = [
        "You are classifying podcast episodes by their FORMAT (structure), "
        "not by their topic, tone, or how dramatic the description sounds.",
        "Choose exactly ONE of these four labels:",
        "- interview: a host speaks with one or more guests in a host-guest, "
        "question-and-answer dynamic.",
        "- solo: a single host speaking alone from memory, experience, or "
        "opinion — no guests and no external sources assembled into the episode "
        "(a gripping first-person story from the host's own memory is still solo).",
        "- panel: three or more speakers (or two co-hosts) discussing as rough "
        "equals, with no single person being interviewed.",
        "- narrative: a story assembled from EXTERNAL sources (reporting, "
        "archives, other people's interviews or recordings) woven into a story arc.",
        "",
    ]

    if labeled_examples:
        lines.append("Here are labeled examples. Learn the pattern from them:")
        lines.append("")
        for ex in labeled_examples:
            lines.append(f"Title: {ex['title']}")
            lines.append(f"Description: {ex['description']}")
            lines.append(f"Label: {ex['label']}")
            lines.append("---")
        lines.append("")

    lines.append("Now classify this new episode:")
    lines.append(f"Description: {description}")
    lines.append("")
    lines.append("Respond in EXACTLY this format and nothing else:")
    lines.append("Label: <one of: interview, solo, panel, narrative>")
    lines.append("Reasoning: <one or two sentences on the structural signal>")

    return "\n".join(lines)


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.

    TODO — Milestone 2 (complete after build_few_shot_prompt):

    Steps:
      1. Call build_few_shot_prompt() to construct the prompt
      2. Send it to the LLM via _client.chat.completions.create()
      3. Parse the response to extract a label and reasoning
      4. Validate the label — if it's not in VALID_LABELS, set it to "unknown"
      5. Return a dict with "label" and "reasoning" keys

    Handle the case where the LLM returns something unparseable gracefully —
    don't let a bad response crash the whole evaluation.

    Before writing code, complete specs/classifier-spec.md.
    """
    prompt = build_few_shot_prompt(labeled_examples, description)

    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
        )
        text = response.choices[0].message.content or ""
    except Exception as exc:
        # Network/API failure — degrade gracefully so the 20-call eval loop
        # in Milestone 3 doesn't crash on a single bad response.
        return {"label": "unknown", "reasoning": f"Classification error: {exc}"}

    return _parse_response(text)


def _parse_response(text: str) -> dict:
    """
    Extract a (label, reasoning) pair from the LLM's raw text response.

    Robust to capitalization, markdown (**interview**), trailing punctuation,
    and the label being on a "Label:" line or loose in the text. Any label not
    in VALID_LABELS becomes "unknown".
    """
    label_line, reasoning = None, text.strip()

    for line in text.splitlines():
        low = line.strip().lower()
        if low.startswith("label"):
            label_line = line.split(":", 1)[1] if ":" in line else line
        elif low.startswith("reasoning"):
            reasoning = (line.split(":", 1)[1] if ":" in line else line).strip()

    # Prefer the explicit Label: line; fall back to scanning the whole text.
    candidate = label_line if label_line is not None else text
    words = re.sub(r"[^a-z]+", " ", candidate.lower()).split()
    label = next((w for w in words if w in VALID_LABELS), "unknown")

    return {"label": label, "reasoning": reasoning}
