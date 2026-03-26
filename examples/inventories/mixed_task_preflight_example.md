# Mixed-Task Inventory Preflight — Worked Example

This example shows how Gradience reduces a 6-adapter, 4-task inventory from 15 candidate pairs to 2 actionable merge candidates.

## Starting inventory

| Adapter | Task | Status |
|---------|------|--------|
| sst2_r16_s42 | SST-2 (sentiment) | eligible |
| sst2_r16_s123 | SST-2 (sentiment) | eligible |
| qnli_r16_s42 | QNLI (question NLI) | eligible |
| qnli_r16_s123 | QNLI (question NLI) | eligible |
| mnli_s42 | MNLI (multi-genre NLI) | eligible |
| rte_s42 | RTE (textual entailment) | eligible |

**6 adapters, 4 tasks, 15 possible pairs.**

All adapters pass source QA (eligible). No weak sources to exclude early. The challenge is in the pair matrix.

## After pair reports

| Pair risk | Count |
|-----------|-------|
| Medium | 11 |
| High | 4 |

Pair-risk alone does not separate actionable from risky. 11 of 15 pairs look structurally plausible (medium risk).

## After task-boundary advisory

| Advisory status | Count |
|----------------|-------|
| Advisory present (cross-task) | **13** |
| Advisory absent (same-task) | **2** |

The advisory partitions the 15-pair matrix into:
- **2 same-task safe pairs** — sst2_s42 × sst2_s123, qnli_s42 × qnli_s123
- **13 cross-task caution pairs** — all others

This is the single most valuable signal. Without the advisory, a practitioner sees 11 medium-risk pairs and no way to distinguish the 2 safe ones from the 9 that will degrade.

## After neighborhoods

Neighborhoods confirm the partition: each task forms its own group. No cross-task neighborhoods are suggested. The pair matrix compresses into 4 isolated task clusters with 2 within-cluster safe pairs.

## Search-space reduction

| Stage | Candidates |
|-------|-----------|
| Starting pool | 15 pairs |
| After QA | 15 (all eligible) |
| After task-boundary advisory | **2 same-task safe pairs** |
| After neighborhoods | 2 priority candidates |

**Reduction: 15 → 2 (87% reduction)**

The preflight pass saved evaluation of 13 cross-task pairs that would all have degraded at least one task.

## What to do next

1. Evaluate the 2 same-task pairs (sst2 × sst2, qnli × qnli) — these are the merge candidates
2. The 13 cross-task pairs should not be explored casually
3. MNLI and RTE adapters are isolated — no safe merge partners in this inventory

## Commands used

```bash
# 1. Source QA (for each adapter)
gradience audit --peft-dir ./sst2_r16_s42 --json > qa/sst2_r16_s42.json

# 2. Pairwise reports (for each pair)
gradience merge-audit \
  --adapter-a ./sst2_r16_s42 --adapter-b ./qnli_r16_s42 \
  --source-a-qa qa/sst2_r16_s42.json --source-b-qa qa/qnli_r16_s42.json \
  --qa-report --emit-report reports/sst2_s42_vs_qnli_s42.json

# 3. Inventory summary
gradience summarize-inventory --qa-dir qa/ --report-dir reports/ \
  --emit-report inventory/summary.json

# 4. Neighborhoods
gradience suggest-neighborhoods --qa-dir qa/ --report-dir reports/ \
  --emit-report inventory/neighborhoods.json
```

## Data source

This example uses real data from the task-advisory observation round (`results/task_advisory_round/large_multitask_pool/`).
