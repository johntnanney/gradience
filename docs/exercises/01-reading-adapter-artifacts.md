# Exercise 1: Reading Adapter Artifacts

The best way to learn Gradience is to treat it less like a theory and more like a diagnostic instrument. You want to develop an eye for what each artifact is telling you.

This exercise uses the curated demo bundle (`examples/demo/`). No GPU required.

## Setup

```bash
pip install gradience
```

## The three adapters

The demo bundle includes three QA artifacts with different profiles:

| Artifact | Profile |
|---|---|
| `examples/demo/qa/code_lora_qa.json` | Clearly decent |
| `examples/demo/qa/chat_lora_qa.json` | Clearly weak |
| `examples/demo/qa/math_lora_qa.json` | No behavioral eval |

## Step 1: Read the artifacts

```bash
for f in examples/demo/qa/*.json; do
  echo "=== $(basename "$f") ==="
  python3 -c "
import json
d = json.load(open('$f'))
a = d['adapter']
s = d['structural_summary']
e = d['eligibility']
b = d['behavioral_summary']
print(f'  adapter:      {a[\"name\"]}  (rank {a[\"rank_nominal\"]}, {a[\"n_layers\"]} layers)')
print(f'  utilization:  {s[\"utilization_mean\"]:.0%} mean, {s[\"utilization_median\"]:.0%} median')
print(f'  rank waste:   {s[\"rank_waste_ratio\"]:.0%}')
print(f'  flags:        {s[\"flags\"] or \"(none)\"}')
print(f'  eval:         {b[\"eval_dataset\"] or \"none\"}')
if b['eval_available']:
    print(f'  score:        {b[\"adapter_score\"]} vs base {b[\"base_score\"]} ({b[\"metric_name\"]})')
    print(f'  beats base:   {b[\"beats_base\"]}')
print(f'  eligibility:  {e[\"status\"]}')
print(f'  reasons:      {e[\"reasons\"]}')
print()
"
done
```

## Step 2: Compare what you see

Look at the three artifacts side by side. Notice:

- **How `eligibility.status` changes.** One is `eligible`, one is `flagged_weak`, one is `unknown_no_behavioral_eval`. These are three different situations, not three points on a scale.

- **How `reasons` differ from `flags`.** Flags are structural observations (low utilization, high rank waste). Reasons are eligibility judgments that combine structural and behavioral evidence. A flagged adapter might still be eligible if it beats the base model behaviorally.

- **Whether low utilization and high rank waste line up with your intuition.** The chat adapter uses 23% of its allocated rank and wastes 77%. The code adapter uses 61% and wastes 39%. Which one sounds like it learned something useful?

- **How much the artifact tells you structurally vs behaviorally.** The math adapter has strong structural numbers (81% utilization, 19% waste) but no behavioral eval. Is that enough to trust it?

## Exercise

For each adapter, write one sentence answering:

1. Is this adapter worth preserving?
2. Why?
3. What evidence supports that answer?

### code_lora (eligible)

Your answer: ___

<details>
<summary>One possible reading</summary>

Worth preserving. It beats the base model on pass@1 (0.34 vs 0.22), uses 61% of its rank capacity, and has no structural flags. Both the structural and behavioral evidence point the same direction.

</details>

### chat_lora (flagged weak)

Your answer: ___

<details>
<summary>One possible reading</summary>

Not worth preserving as-is. It underperforms the base model on perplexity (5.47 vs 4.66), wastes 77% of its rank, and is flagged for low utilization. The structural and behavioral evidence agree: this adapter learned very little and what it learned made things worse.

</details>

### math_lora (no behavioral eval)

Your answer: ___

<details>
<summary>One possible reading</summary>

Cannot determine. The structural numbers look healthy (81% utilization, no flags), but structural health alone does not guarantee behavioral quality. This adapter needs a downstream evaluation before you can make a real decision. That is exactly what the `unknown_no_behavioral_eval` status means.

</details>

## The point

The core move in Gradience is separating **measurement** from **judgment**.

The structural summary is measurement. The eligibility status is judgment. The behavioral summary is the bridge between them.

When you read a QA artifact, you are not looking for a single number that says "good" or "bad." You are looking at what kind of evidence is present, what it says, and what is missing.

That is the skill this exercise builds.

## Next

- [Exercise 2: Reading Merge-Risk Reports](02-reading-merge-reports.md) -- compare a safe pair and a risky pair
- [Exercise 3: Reading an Inventory Summary](03-reading-inventory-summaries.md) -- what does the whole collection look like?
