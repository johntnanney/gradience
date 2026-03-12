# Exercise 3: Reading an Inventory Summary

This exercise builds on [Exercise 1](01-reading-adapter-artifacts.md) and [Exercise 2](02-reading-merge-reports.md). You now know how to read individual adapters and pairs. The question becomes: what does the whole collection look like?

## The inventory

The demo bundle includes an inventory summary built from all three adapters and both pairs:

```bash
python3 -c "
import json
d = json.load(open('examples/demo/inventory_summary.json'))
s = d['sources']
print(f'Sources: {s[\"qa_artifact_count\"]} adapters, {s[\"merge_report_count\"]} pairs')
print()
for section in ['adapter_status_counts', 'adapter_flag_counts', 'pair_risk_counts',
                'recommended_strategy_counts', 'dominant_issue_counts']:
    counts = d.get(section, {})
    if counts:
        label = section.replace('_counts', '').replace('_', ' ')
        print(f'{label}:')
        for k, v in counts.items():
            print(f'  {k}: {v}')
        print()
print(f'strict-QA block candidates: {d[\"strict_qa_block_candidates\"]}')
"
```

## What you should see

```
Sources: 3 adapters, 2 pairs

adapter status:
  eligible: 1
  flagged_weak: 1
  unknown_no_behavioral_eval: 1

adapter flag:
  low_utilization: 1
  high_rank_waste: 1

pair risk:
  high: 1
  low: 1

recommended strategy:
  audit_aware: 1
  linear: 1

dominant issue:
  none: 1
  subspace_conflict: 1

strict-QA block candidates: 2
```

## Step 1: Read the counts

The inventory summary is a set of counters. It tells you the shape of your collection without requiring you to open every individual artifact.

Notice:

- **Adapter status distribution.** One-third eligible, one-third weak, one-third unverified. That is not a healthy inventory. In a real scenario, you would want most adapters to be eligible before proceeding with merges.

- **Flags vs status.** Only 1 adapter has structural flags, but 2 adapters have problematic eligibility status. Structural flags are a subset of the reasons an adapter might not be ready.

- **Pair risk distribution.** Half low-risk, half high-risk. With only 2 pairs this is not very informative, but in a real 10-adapter inventory with 45 possible pairs, the risk distribution quickly becomes the most important summary.

- **Strict-QA block candidates: 2.** Both pairs would be blocked under `--strict-qa`, because each pair includes at least one adapter that is either `flagged_weak` or `unknown_no_behavioral_eval`. That is the inventory telling you: none of these merges are cleared for a strict workflow.

## Exercise

Looking at just the inventory summary (not the individual artifacts), answer:

1. Is this inventory ready for production merges?
2. What is the single most important action to take next?
3. How many adapters need work before any merge can proceed under strict-QA?

Your answers: ___

<details>
<summary>One possible reading</summary>

1. No. Two of three adapters are not eligible, and both pairs are blocked under strict-QA.

2. Run behavioral evaluations on the math adapter. It has strong structural numbers but no eval -- that is the cheapest path to clearing one adapter and potentially unblocking the code+math pair.

3. Two. The chat adapter needs to either be retrained or dropped (it underperforms the base model). The math adapter needs a behavioral evaluation. Only after both are addressed can any pair clear strict-QA.

</details>

## The point

The inventory summary is not a dashboard. It is a triage tool.

It answers: given the current state of my adapter collection, where are the obvious problems and what should I look at first?

That is useful precisely because it is descriptive, not prescriptive. It counts what you have. It does not tell you what to do about it. That judgment is still yours.

## What you have learned

Across these three exercises:

1. **Exercise 1** -- How to read a single adapter's structural and behavioral evidence, and how eligibility judgment separates measurement from decision.

2. **Exercise 2** -- How to read a pair's geometric compatibility, what dominant issues mean operationally, and how source quality flows into merge recommendations.

3. **Exercise 3** -- How to read an inventory as a triage summary, identify the weakest links, and decide where to invest effort next.

That is the Gradience diagnostic skill: reading artifacts, understanding what evidence is present, and knowing what is missing.
