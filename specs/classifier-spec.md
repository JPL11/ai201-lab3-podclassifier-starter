# Classifier Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 2.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `build_few_shot_prompt()` and
`classify_episode()` in `classifier.py`.

---

## build_few_shot_prompt(labeled_examples, description)

### What it does
Constructs a prompt string for the LLM that includes the task instructions,
all labeled training examples, and the new episode description to classify.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `labeled_examples` | `list[dict]` | Each dict has `"title"`, `"description"`, `"label"` (and others). These are the examples you labeled in Milestone 1. |
| `description` | `str` | The episode description to classify. |

### Output

| Return value | Type | Description |
|---|---|---|
| prompt | `str` | A complete prompt string ready to send to the LLM. |

---

### Spec fields — fill these in before writing code

**Task instruction (what should the LLM know about the task?):**

```
You are classifying podcast episodes by their format. Classify the episode
into exactly one of these four labels:

- interview: a conversation between a host and one or more guests
- solo: a single host speaking from memory, experience, or opinion — no guests,
  no assembled external sources
- panel: multiple guests with roughly equal speaking time, often debating or
  discussing a topic together
- narrative: a story assembled from external sources — interviews, archival
  audio, reporting — with a clear narrative arc

Return only the label and your reasoning. Do not explain the taxonomy.
```

---

**How should labeled examples be formatted in the prompt?**

```
Each example should include the episode title, a brief excerpt or the full
description, and the correct label. Separate examples with a blank line or
a delimiter like "---". Include all fields that help the model see why the
label was applied — title and description are both useful; other fields
(like episode ID) are not needed.
```

---

**Example block sketch (write one concrete example):**

```
Title: {title}
Description: {description}
Label: {label}
```

---

**How should the new episode (to be classified) be presented?**

```
Present it in the same format as the labeled examples, but omit the Label
line and replace it with an instruction to classify. For example:

Title: {title}
Description: {description}
Label: ?

Then add a line like: "Classify the episode above. Return your answer in
the format below:" followed by the output format you chose.
```

---

**What output format should you request from the LLM?**

```
A two-line, prefixed format:

    Label: <one of interview, solo, panel, narrative>
    Reasoning: <one or two sentences>

Tradeoffs considered:
- JSON: cleanest in theory, but llama-3.3 often wraps it in ```json fences or
  adds prose around it, so json.loads() fails and you're back to regex anyway.
- Bare label on its own line: easy, but throws away the reasoning the spec
  asks for, and the model tends to add a sentence regardless.
- "Label:/Reasoning:" prefixes (chosen): trivially parseable (scan for the
  line starting with "label", take what's after the colon), human-readable,
  and the explicit "Label" anchor lets parsing ignore any extra prose. Put
  Label FIRST so even a truncated response still yields a label.
```

---

**Edge cases to handle in the prompt:**

```
- labeled_examples empty: skip the "Here are labeled examples" block entirely
  and fall back to a zero-shot prompt (task instructions + the four label
  definitions + the description). The label definitions carry the signal when
  there are no demonstrations.
- Very short description: still presented the same way; the four label
  definitions in the instructions give the model enough to make a structural
  guess. If it genuinely can't tell, the validation layer downstream will
  catch an off-taxonomy answer as "unknown" rather than guessing wildly.
```

---

## classify_episode(description, labeled_examples)

### What it does
Classifies a single podcast episode description using the few-shot LLM classifier.
Returns a dict with a label and reasoning.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `description` | `str` | The episode description to classify. |
| `labeled_examples` | `list[dict]` | Labeled training examples from `load_labeled_examples()`. |

### Output

| Return value | Type | Description |
|---|---|---|
| result | `dict` | Must have keys `"label"` and `"reasoning"`. `"label"` must be one of `VALID_LABELS` or `"unknown"`. |

---

### Spec fields — fill these in before writing code

**Step 1 — Build the prompt:**

```
Call build_few_shot_prompt(labeled_examples, description) and store the
returned string in a variable (e.g., prompt). Pass through both arguments
exactly as received — no modification needed before calling.
```

---

**Step 2 — Send to the LLM:**

```
Call _client.chat.completions.create() with:
  - model: the model name from config (MODEL_NAME)
  - messages: a list with one dict — {"role": "user", "content": prompt}
  - max_tokens: a reasonable limit (e.g., 200–300) to keep responses concise

Extract the response text from:
  response.choices[0].message.content
```

---

**Step 3 — Parse the response:**

```
Split the text into lines. Find the line that starts with "label" (case-
insensitive) and take everything after the first colon; do the same for
"reasoning". Then NORMALIZE the label candidate: lowercase it, strip non-letter
characters (handles "**Interview**", "Label: narrative.", etc.), split into
words, and take the first word that is in VALID_LABELS. If no Label: line is
found, scan the whole response for the first valid-label word as a fallback.
```

---

**Step 4 — Validate the label:**

```
The normalization in step 3 IS the validation: a candidate only becomes the
label if it matches an entry in VALID_LABELS. If nothing matches (the model
invented a label like "storytelling", or the response was empty/truncated),
set label to "unknown". Never return a label outside VALID_LABELS ∪ {unknown}.
```

---

**Step 5 — Handle errors gracefully:**

```
Wrap the API call in try/except. On ANY exception — network error, rate limit
(HTTP 429), timeout, malformed response — return
{"label": "unknown", "reasoning": f"Classification error: {exc}"} instead of
raising. The evaluation loop makes 20 calls; one failure must not abort the
whole run. (Note: this means a 429 surfaces as "unknown" in the report — when
debugging a cluster of unknowns, check the reasoning field to tell a parse
failure apart from an infrastructure error.)
```

---

### Return value structure

```python
{
    "label": str,      # one of VALID_LABELS, or "unknown" if invalid/error
    "reasoning": str,  # brief explanation from the LLM
}
```

---

## Notes on label quality

The classifier is only as good as your labels. If your training examples have
inconsistent or ambiguous labels, the LLM will learn the wrong pattern.

Before implementing the classifier, re-read `data/taxonomy.md` and double-check
any labels you're unsure about. Annotation quality is part of the lab.

---

## Implementation Notes

*Fill this in after implementing and testing both functions.*

**Test: what does the raw LLM response look like for one episode?**

```
The model follows the requested format closely, e.g.:

    Label: interview
    Reasoning: The host speaks with one guest (Dr. Priya Nair) in a clear
    question-and-answer dynamic, drawing out her expertise.

(Exact raw text to be re-captured on the next run — the diagnostic call to
grab it hit Groq's free-tier daily token cap; see the unknown note below.)
```

**How did you parse the label out of the response?**

```
Split on lines, grab the text after "Label:", lowercase it, strip non-letters
with regex, and take the first word found in VALID_LABELS. Unit-tested against
"Interview", "**Interview**", "Label: Narrative.", and "The format is: solo." —
all parse correctly; "storytelling vibes" correctly yields "unknown".
```

**Did any episodes return `"unknown"`? If so, why?**

```
Yes — 3 of the 5 narrative test episodes (the LAST three processed). NOT a
parsing or model failure: interview/solo/panel scored 100% (15/15) and 2
narrative episodes classified correctly. The 3 unknowns were the final calls in
the 20-call eval, which hit Groq's free-tier daily token limit (100k tokens/
day). classify_episode() catches the 429 and returns "unknown" by design. A
re-run after the limit resets should put narrative at ~4–5/5.
```

**One thing about the output format that surprised you:**

```
How cleanly "Label: first, Reasoning: second" survives real-world noise. The
fragile part turned out not to be the format at all — it was conflating an
infrastructure error (rate limit) with a parse failure, because both collapse
to "unknown". The fix isn't in the parser; it's reading the reasoning field to
distinguish the two.
```
