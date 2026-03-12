# Exercise 2: Reading Merge-Risk Reports

This exercise builds on [Exercise 1](01-reading-adapter-artifacts.md). You now know what individual adapters look like. The question becomes: what happens when you pair them?

## The two pairs

The demo bundle includes two merge-risk reports:

| Report | Pairing |
|---|---|
| `examples/demo/reports/code_math_safe.json` | eligible + missing-QA |
| `examples/demo/reports/code_chat_risky.json` | eligible + flagged weak |

## Step 1: Read the reports

```bash
for f in examples/demo/reports/*.json; do
  echo "=== $(basename "$f") ==="
  python3 -c "
import json
d = json.load(open('$f'))
a, b = d['adapter_a'], d['adapter_b']
v = d['verdict_distribution']
print(f'  A: {a[\"path\"]} (rank {a[\"rank\"]}, {a[\"eligibility_status\"]})')
print(f'  B: {b[\"path\"]} (rank {b[\"rank\"]}, {b[\"eligibility_status\"]})')
print(f'  risk:       {d[\"pair_risk\"]}')
print(f'  issue:      {d[\"dominant_issue\"]} -- {d[\"dominant_issue_detail\"]}')
print(f'  strategy:   {d[\"recommended_strategy\"]}')
print(f'  confidence: {d[\"confidence\"]}')
print(f'  score:      {d[\"compatibility_score\"]}')
print(f'  verdicts:   {v[\"safe\"]} safe, {v[\"redundant\"]} redundant, {v[\"conflicting\"]} conflicting, {v[\"imbalanced\"]} imbalanced')
if d.get('caveats'):
    print(f'  caveats:')
    for c in d['caveats']:
        print(f'    - {c}')
print()
"
done
```

## Step 2: Compare what you see

Notice:

- **Risk level is not just a number.** The safe pair has a compatibility score of 0.874 and the risky pair has 0.298, but what matters more is the *kind* of risk. The risky pair has 14 conflicting layers and 6 imbalanced layers. The safe pair has 28 safe layers and only minor redundancy.

- **Dominant issue tells you what to worry about.** The safe pair's dominant issue is `none`. The risky pair's is `subspace_conflict`. That label is the single most useful field for deciding what to do next.

- **Strategy follows from risk.** Low risk gets `linear` (simple, cheap). High risk gets `audit_aware` (per-layer, more careful). This is not arbitrary -- it reflects how much structural adjustment the pair needs.

- **Eligibility status flows through.** The safe pair's adapter B is `unknown_no_behavioral_eval`. The risky pair's adapter B is `flagged_weak`. Those came from the QA artifacts. The merge report inherits them and adjusts confidence accordingly.

- **Caveats appear only when warranted.** The safe pair has one caveat (missing behavioral eval). The risky pair has three (weak adapter, over-provisioned layers, high structural risk). Caveats are the report's way of saying "here is what could go wrong even if you follow the recommendation."

## Exercise

For each pair, answer:

1. Would you proceed with this merge?
2. What is the single strongest piece of evidence for your decision?
3. What would change your mind?

### code + math (safe pair)

Your answer: ___

<details>
<summary>One possible reading</summary>

Probably yes, with caution. The structural compatibility is strong (28/32 layers safe, score 0.874). The main risk is that the math adapter has no behavioral eval -- so structural safety does not guarantee the merge produces a good model. You would want to run a downstream eval on the merged result. What would change your mind: if the math adapter turns out to perform poorly on its own task.

</details>

### code + chat (risky pair)

Your answer: ___

<details>
<summary>One possible reading</summary>

Probably no, or at least not without significant investigation. The chat adapter already underperforms the base model, 14 layers are in subspace conflict, and confidence is low. Merging a strong adapter with a weak one that fights it structurally is unlikely to produce something better than the strong adapter alone. What would change your mind: evidence that the chat adapter contributes something the code adapter genuinely lacks, despite the perplexity regression.

</details>

## The point

A merge-risk report is not a go/no-go signal. It is a structured summary of what the pairing looks like structurally, what the source quality situation is, and what kind of merge strategy the geometry suggests.

The decision is still yours. The report's job is to make sure you are not surprised.

## Next

- [Exercise 3: Reading an Inventory Summary](03-reading-inventory-summaries.md) -- what does the whole collection look like?
