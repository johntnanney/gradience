# Same-Task Control Inventory — Worked Example

This example shows what Gradience looks like when the inventory is already clean: all adapters are on the same task, all pass QA, and the workflow is mostly confirmatory.

## Starting inventory

| Adapter | Task | Rank | Status |
|---------|------|------|--------|
| qnli_r16_s42 | QNLI | 16 | eligible |
| qnli_r16_s123 | QNLI | 16 | eligible |
| qnli_r8_s42 | QNLI | 8 | eligible |

**3 adapters, 1 task, 3 possible pairs.**

## After pair reports

| Pair risk | Count |
|-----------|-------|
| Medium | 3 |

All 3 pairs rated medium risk (structural variation from different ranks and seeds, but no concerning issues).

## Task-boundary advisory

**Advisory is silent on all 3 pairs.** This is expected and correct — all adapters share the same task.

## Interpretation

This is a same-task safe pool. On small encoder models, same-task pairs are broadly safe across all tested stressors (training style, domain shift, source-strength asymmetry — 49 pairs, 0 material degradations in validation studies).

The workflow confirms what context already suggests: all 3 pairs are reasonable merge candidates.

## Search-space reduction

| Stage | Candidates |
|-------|-----------|
| Starting pool | 3 pairs |
| After QA | 3 (all eligible) |
| After advisory | 3 (all same-task, advisory silent) |

**Reduction: none needed.** The inventory is already clean.

## When this happens

This is the low-value regime for Gradience. When you already know:
- all adapters are on the same task
- all pass basic quality checks
- the pool is small

...the full preflight workflow is confirmatory rather than decision-changing. The main value is the explicit QA record, not the narrowing.

## Data source

This example uses real data from the advisory observation round (`results/task_advisory_round/same_task_qnli_control/`).
