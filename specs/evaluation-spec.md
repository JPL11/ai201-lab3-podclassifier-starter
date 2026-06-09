# Evaluation Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:
- **Overall:** What fraction of episodes did we classify correctly?
- **Per-class:** Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What it does
Returns the fraction of predictions that exactly match the ground truth.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`, one per episode. |
| `ground_truth` | `list[str]` | The correct labels, in the same order as `predictions`. |

### Output

| Return value | Type | Description |
|---|---|---|
| accuracy | `float` | A value between 0.0 and 1.0. |

---

### Spec fields — fill these in before writing code

**Formula:**

```
accuracy = (number of predictions that exactly equal the ground-truth label
in the same position) / (total number of predictions). A prediction is
"correct" only on an exact string match (predicted == truth).
```

---

**Step-by-step logic:**

```
1. If ground_truth is empty, return 0.0 (avoid divide-by-zero).
2. Walk predictions and ground_truth together with zip(), counting how many
   pairs are equal.
3. Divide that count by len(ground_truth) and return the float.
```

---

**Edge case — what if both lists are empty?**

```
Return 0.0. There are no correct predictions and dividing by zero is undefined,
so 0.0 is the safe, meaningful answer (no measured accuracy).
```

---

**Worked example:**

```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

Position 0: interview == interview  ✓
Position 1: solo      == solo        ✓
Position 2: panel     != solo        ✗
Position 3: interview != narrative   ✗
correct = 2, total = 4  →  2/4 = 0.5
```

---

## compute_per_class_accuracy(predictions, ground_truth)

### What it does
Returns accuracy broken down by each label. For each label in `VALID_LABELS`,
reports how many episodes with that ground-truth label were classified correctly.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`. |
| `ground_truth` | `list[str]` | Correct labels, in the same order. |

### Output

A `dict` keyed by label. Each value is a dict with three keys:

```python
{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}
```

---

### Spec fields — fill these in before writing code

**What does "correct" mean for a given class?**

```
For class C, an episode is "correct" when its GROUND-TRUTH label is C AND the
prediction also equals C. Grouping is by ground truth, so a wrong prediction
(e.g. truth=panel, pred=interview) counts against panel's accuracy, never
interview's. (Predicting C when the truth is something else is a different
class's error — it doesn't help class C's score.)
```

---

**What does "total" mean for a given class?**

```
The number of episodes whose GROUND-TRUTH label is C — i.e. how many real C
episodes exist in the test set. Not the number of times C was predicted. This
is the denominator: "of all the true C episodes, how many did we get right?"
(This is recall per class.)
```

---

**Step-by-step logic:**

```
1. Initialize a dict: {label: {"correct": 0, "total": 0, "accuracy": 0.0}}
   for every label in VALID_LABELS.
2. Loop over (pred, truth) pairs with zip().
3. For each pair, if truth is a valid label: increment stats[truth]["total"],
   and if pred == truth also increment stats[truth]["correct"].
4. After the loop, set each class's "accuracy" = correct / total, or 0.0 when
   total == 0.
5. Return the dict.
```

---

**Edge case — what if a class has no examples in ground_truth (total == 0)?**

```
Set accuracy to 0.0 (and leave correct/total at 0). Dividing by zero is
undefined, and there's nothing to be right about, so 0.0 is the convention the
evaluate.py docstring specifies.
```

---

**Worked example:**

```
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

Grouping by ground truth:
- interview: 1 true (pos 0), predicted interview ✓        → 1/1
- solo:      2 true (pos 1,2); pos1 pred interview ✗, pos2 pred solo ✓ → 1/2
- panel:     1 true (pos 3), predicted panel ✓            → 1/1
- narrative: 1 true (pos 4), predicted panel ✗            → 0/1

label       correct  total  accuracy
----------  -------  -----  --------
interview      1       1      1.0
solo           1       2      0.5
panel          1       1      1.0
narrative      0       1      0.0
```

---

## Reflection questions (discuss at the checkpoint)

1. Your overall accuracy might be decent even if one class has very low accuracy.
   Why is per-class accuracy a more informative metric than overall accuracy alone?

2. If `panel` episodes consistently get misclassified as `interview`, what does
   that tell you about your training labels or your prompt?

3. You labeled 20 training episodes and evaluated on 20 test episodes (5 per class).
   How might the evaluation results change if you had labeled 100 training episodes?
   What if you had 200 test episodes?
