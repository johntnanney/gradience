# Mixed-Task Inventory Walkthrough

This walkthrough shows Gradience reducing a 6-adapter, 4-task inventory from 15 candidate pairs to 2.

## Starting inventory

| Adapter | Task | Rank |
|---------|------|------|
| sst2_r16_s42 | SST-2 (sentiment) | 16 |
| sst2_r16_s123 | SST-2 (sentiment) | 16 |
| qnli_r16_s42 | QNLI (question NLI) | 16 |
| qnli_r16_s123 | QNLI (question NLI) | 16 |
| mnli_s42 | MNLI (multi-genre NLI) | 16 |
| rte_s42 | RTE (textual entailment) | 16 |

**6 adapters. 4 tasks. 15 possible pairs.** Which ones are worth evaluating?

---

## Step 1: Source QA

```bash
for adapter in sst2_r16_s42 sst2_r16_s123 qnli_r16_s42 qnli_r16_s123 mnli_s42 rte_s42; do
  gradience audit --peft-dir ./adapters/$adapter --json > qa/${adapter}.json
done
```

**Result:** All 6 adapters are eligible. No weak sources to exclude.

In a messier pool, QA would typically remove 1-2 adapters here — that alone cuts the pair matrix substantially. In this inventory, all sources are credible, so the narrowing has to come from the pair layer.

---

## Step 2: Pairwise merge reports

```bash
# Run all 15 pairs (example for one pair)
gradience merge-audit \
  --adapter-a ./adapters/sst2_r16_s42 \
  --adapter-b ./adapters/qnli_r16_s42 \
  --source-a-qa qa/sst2_r16_s42.json \
  --source-b-qa qa/qnli_r16_s42.json \
  --qa-report --emit-report reports/sst2_s42_vs_qnli_s42.json
```

**Result: 15 pair reports.** Here is the full matrix:

| Pair | Risk | Advisory | Issue |
|------|------|----------|-------|
| sst2_s42 × sst2_s123 | medium | — | high redundancy |
| qnli_s42 × qnli_s123 | medium | — | partial redundancy |
| sst2_s42 × qnli_s42 | **high** | **YES** | norm imbalance |
| sst2_s42 × qnli_s123 | **high** | **YES** | norm imbalance |
| sst2_s42 × mnli_s42 | medium | **YES** | partial redundancy |
| sst2_s42 × rte_s42 | **high** | **YES** | subspace conflict |
| sst2_s123 × qnli_s42 | **high** | **YES** | norm imbalance |
| sst2_s123 × qnli_s123 | medium | **YES** | partial redundancy |
| sst2_s123 × mnli_s42 | medium | **YES** | partial redundancy |
| sst2_s123 × rte_s42 | medium | **YES** | partial redundancy |
| qnli_s42 × mnli_s42 | medium | **YES** | norm imbalance |
| qnli_s42 × rte_s42 | medium | **YES** | partial redundancy |
| qnli_s123 × mnli_s42 | medium | **YES** | norm imbalance |
| qnli_s123 × rte_s42 | medium | **YES** | partial redundancy |
| mnli_s42 × rte_s42 | medium | **YES** | partial redundancy |

### Reading the matrix

Pair-risk alone is not enough: 11 of 15 pairs are medium risk. That leaves too many candidates.

The advisory resolves this. **13 of 15 pairs carry a task-boundary warning.** Only 2 are advisory-free:
- sst2_s42 × sst2_s123 (same task)
- qnli_s42 × qnli_s123 (same task)

**Result: 2 same-task safe pairs, 13 cross-task caution pairs.** Without the advisory, a practitioner sees 11 plausible-looking candidates. With the advisory, the answer is immediate.

---

## Step 3: Inventory summary and neighborhoods

```bash
gradience summarize-inventory --qa-dir qa/ --report-dir reports/ \
  --emit-report inventory/summary.json

gradience suggest-neighborhoods --qa-dir qa/ --report-dir reports/ \
  --emit-report inventory/neighborhoods.json
```

**Neighborhoods confirm the partition.** Each task forms its own isolated group. No cross-task neighborhoods are suggested. The pair matrix compresses into 4 task clusters with within-cluster safe pairs only.

---

## Step 4: Interpret and reduce

The preflight pass produces a clear decision surface:

| Zone | Pairs | Action |
|------|-------|--------|
| **Same-task safe** | sst2_s42 × sst2_s123, qnli_s42 × qnli_s123 | Evaluate these — they are your merge candidates |
| **Cross-task caution** | 13 pairs | Do not prioritize — all cross-task merges degrade at least one task |
| **Isolated adapters** | mnli_s42, rte_s42 | No safe merge partners in this inventory |

---

## Before and after

| Stage | Candidate pairs |
|-------|----------------|
| Starting pool | **15** |
| After source QA | 15 (all eligible) |
| After task-boundary partitioning | **2** same-task safe pairs |
| After neighborhood confirmation | **2** priority candidates |

**Search-space reduction: 87%.** The preflight saved evaluation of 13 cross-task pairs. In the utility round across 5 inventories, comparable mixed-task pools saw 65-90% reduction, with 81% average where the advisory was the main discriminator.

---

## Step 5: Evaluate the reduced set

Spend evaluation budget on:
1. sst2_s42 × sst2_s123
2. qnli_s42 × qnli_s123

These are the only pairs in this inventory where a linear merge is likely to preserve both adapters' task performance.

---

## Data source

This walkthrough uses real data. Artifacts are available in two locations:
- `examples/inventory_preflight_mixed_task/` — self-contained example bundle
- `results/task_advisory_round/large_multitask_pool/` — original execution output
