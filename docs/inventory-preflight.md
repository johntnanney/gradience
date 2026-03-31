# Inventory Preflight Workflow

## What Gradience is for

You have a pool of LoRA adapters. Some might merge well. Most probably won't. Gradience runs a preflight pass that:

- **Reduces the search space** — in utility testing, mixed-task inventories saw 65-90% candidate reduction before any behavioral evaluation
- **Exposes task-boundary risk** — identifies cross-task pairs where structural similarity is misleading (0 false positives across 132+ checks)
- **Partitions the inventory** — separates same-task safe zones from cross-task caution zones
- **Saves evaluation budget** — you evaluate the reduced set, not the full pair matrix

The output is not a merge recommendation. It is a narrower, more defensible set of candidates worth testing.

## Quickstart

```bash
# 1. Screen each adapter
for d in ./adapters/*/; do
  name=$(basename "$d")
  gradience audit --peft-dir "$d" --json > qa/${name}.json
done

# 2. Run all pairwise reports (example for one pair)
gradience merge-audit \
  --adapter-a ./adapters/adapter_1 --adapter-b ./adapters/adapter_2 \
  --source-a-qa qa/adapter_1.json --source-b-qa qa/adapter_2.json \
  --qa-report --emit-report reports/adapter_1_vs_adapter_2.json

# 3. Summarize the inventory
gradience summarize-inventory --qa-dir qa/ --report-dir reports/ \
  --emit-report inventory/summary.json
gradience suggest-neighborhoods --qa-dir qa/ --report-dir reports/ \
  --emit-report inventory/neighborhoods.json

# 4. Read the outputs
#    - Pairs WITHOUT task_relationship_advisory → same-task safe zone
#    - Pairs WITH advisory → cross-task caution zone
#    See the mixed-task walkthrough for a worked example.
```

## When to use

**High value:**
- 5+ adapters from mixed tasks or sources
- Pair matrices too dense to reason about manually (15+ pairs)
- Mixed provenance — some adapters well-tested, others not
- You want to reduce the number of merges you actually evaluate

**Lower value:**
- 3 same-task adapters that are all known to work
- The decision is already obvious from context
- You only have one pair to evaluate

## Standard workflow

### Step 1: Screen source quality

```bash
gradience audit --peft-dir ./adapter_1 --json > qa/adapter_1.json
# repeat for each adapter
```

This identifies weak, under-evidenced, or structurally problematic adapters. In messy pools, source QA alone often removes the most problematic candidates before pairwise analysis begins.

**What to look for:** `eligibility` status in the QA artifact. Adapters marked `flagged_weak` or `unknown_no_behavioral_eval` should be treated cautiously.

**Evidence tiers.** Each adapter's behavioral evidence falls into one of three tiers:

- **`behavioral_reported`** — evaluation data is present and the adapter is eligible or uncertain. This is the strongest tier, but "reported" means user-provided: Gradience does not independently verify claimed evaluation results.
- **`behavioral_weak`** — evaluation data is present but the adapter underperforms its base model (`flagged_weak`). The evidence exists but points against the adapter.
- **`behavioral_missing`** — no evaluation data was provided (`unknown_no_behavioral_eval`). Structural analysis is the only available signal.

The distinction matters because "weak" and "missing" are different failure modes. A weak adapter has been tested and found wanting — you know what you're dealing with. A missing-evidence adapter is an unknown — the structural audit may look fine, but there is no behavioral ground truth to anchor the recommendation. The inventory summary and action plan both surface these tiers so you can calibrate trust accordingly.

### Step 2: Run pairwise merge reports

```bash
gradience merge-audit \
  --adapter-a ./adapter_1 --adapter-b ./adapter_2 \
  --source-a-qa qa/adapter_1.json --source-b-qa qa/adapter_2.json \
  --qa-report --emit-report reports/pair_1_2.json
# repeat for each pair
```

Each report includes:
- **pair_risk**: low / medium / high
- **dominant_issue**: what kind of structural concern exists
- **task_relationship_advisory**: present when adapters were evaluated on different tasks

### Step 3: Run inventory summary and neighborhoods

```bash
gradience summarize-inventory \
  --qa-dir qa/ --report-dir reports/ \
  --emit-report inventory/summary.json

gradience suggest-neighborhoods \
  --qa-dir qa/ --report-dir reports/ \
  --emit-report inventory/neighborhoods.json
```

The summary aggregates counts across all adapters and pairs. Neighborhoods compress the pair matrix into interpretable groups.

### Step 4: Interpret and reduce

Read the outputs as a decision surface:

1. **Same-task safe zone** — pairs where both adapters share the same evaluation task and the advisory is silent. These are your best merge candidates.
2. **Cross-task caution zone** — pairs where the advisory fires. On small encoder models, all cross-task merges degrade at least one task. Proceed only with specific justification.
3. **Excluded adapters** — weak or unknown-provenance adapters that QA identified early.
4. **Reduced candidate set** — the pairs remaining after QA exclusion and task-boundary partitioning.

### Step 5: Evaluate only the reduced set

Spend evaluation budget on the narrowed candidate set, not the full pair matrix.

## How each signal contributes

| Signal | Role | Strongest when |
|--------|------|---------------|
| Source QA | First anchor — removes weak/unknown adapters | Messy pools with mixed provenance |
| Pair-risk | Default structural layer — flags geometric issues | All regimes |
| Task advisory | Boundary detection — partitions same-task from cross-task | Mixed-task inventories (highest value) |
| Neighborhoods | Inventory compression — groups pairs into interpretable regions | 6+ adapters where pair matrix is dense |

## What the advisory means

When the `task_relationship_advisory` is present on a pair report, it means the two adapters were evaluated on different tasks. This is the most important practical warning in mixed-task inventories:

- **Advisory present** — cross-task pair. On small encoder models, these consistently degrade at least one task. Do not prioritize unless you have a specific reason.
- **Advisory absent** — same-task pair. These are broadly safe across all tested stressors (training style, domain, source strength — 49 pairs, 0 material degradations).

The advisory does not grade severity within cross-task pairs. It catches the boundary; what happens inside that boundary varies by backbone and task pair.

## What a good outcome looks like

After a preflight pass, you should have:

- **Weak sources excluded** — before they distort pairwise analysis
- **Cross-task region deprioritized** — moved to a caution list, not your evaluation queue
- **Same-task region prioritized** — these are your merge candidates
- **Candidate count reduced** — in utility testing across 5 inventories, 65-90% of pairs were eliminated (81% average where the advisory was the main discriminator)

If the inventory is same-task and clean, the preflight will confirm that — and that confirmation is also useful, because it means no hidden task-boundary risk exists.

## Further reading

- **[Playbook](playbook.md)** — step-by-step instructions for the five most common workflows
- **[Product Validation](product-validation.md)** — field trial evidence: what Gradience gets right, where the limits are
- **[Example Gallery](example-gallery.md)** — six canonical scenarios: same-task, mixed-task, large inventory, weak-evidence, near-miss, retained-vs-control evaluation
- **[Mixed-task inventory walkthrough](examples/mixed-task-inventory-walkthrough.md)** — flagship: 6 adapters, 4 tasks, 15 pairs reduced to 2
- **[Same-task control walkthrough](examples/same-task-control-walkthrough.md)** — contrast case: advisory silence, confirmatory behavior

## What Gradience will and will not tell you

**It will tell you:**
- Which adapters are structurally weak or lack behavioral evidence
- Which pairs cross a task boundary (and should not be merged casually)
- Which same-task pairs are your safest merge candidates
- How to reduce a 15-pair matrix to 2-4 pairs worth evaluating

**It will not tell you:**
- How much a cross-task merge will degrade. The advisory flags cross-task pairs, but whether a flagged pair degrades by 2pp or 40pp depends on the backbone and exact task combination. No current signal grades this reliably across backbones.
- Whether a merge will actually improve your downstream metric. Preflight narrows the candidates; evaluation still decides the outcome.
- Anything about decoder-only or large-scale models. Current evidence is on small encoder models (DistilBERT, RoBERTa).
