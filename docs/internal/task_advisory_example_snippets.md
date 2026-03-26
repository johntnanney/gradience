# Task Advisory — Example Report Snippets

Reusable snippets for blog posts, documentation, and walkthroughs.

## Example 1: Merge report WITHOUT advisory (same-task pair)

```
  MERGE QA REPORT
  ============================================================

  1. STRUCTURAL RESULT
  ----------------------------------------
  Pair risk:       MEDIUM
  Compat. score:   0.805
  Dominant issue:  HIGH REDUNDANCY
                   24 redundant layer(s)
  Layer verdicts:  24 redundant (24 total)

  Adapter A:  rank=16, alpha=16.0, 24 layers (distilbert-base-uncased)
              results/adapters/qnli_s42
  Adapter B:  rank=16, alpha=16.0, 24 layers (distilbert-base-uncased)
              results/adapters/qnli_s7

  2. BEHAVIORAL STATUS
  ----------------------------------------
  Adapter A eligibility: eligible
  Adapter B eligibility: eligible

  Confidence:      high
                   High spectral compatibility (score=0.805) — both adapters have verified behavioral quality

  3. ELIGIBILITY WARNING
  ----------------------------------------
  No eligibility concerns detected.

  4. RECOMMENDED ACTION
  ----------------------------------------
  Merge is safe. Use audit-aware strategy or norm-equalized baseline.
  Strategy: linear
```

No advisory section appears. Both adapters evaluated on `qnli_dev`. Same-task pair — merge is safe.

---

## Example 2: Merge report WITH advisory (cross-task pair)

```
  MERGE QA REPORT
  ============================================================

  1. STRUCTURAL RESULT
  ----------------------------------------
  Pair risk:       MEDIUM
  Compat. score:   0.313
  Dominant issue:  PARTIAL REDUNDANCY
                   13 redundant layer(s)
  Layer verdicts:  11 safe, 13 redundant (24 total)

  Adapter A:  rank=16, alpha=16.0, 24 layers (distilbert-base-uncased)
              results/adapters/qnli_s42
  Adapter B:  rank=16, alpha=16.0, 24 layers (distilbert-base-uncased)
              results/adapters/rte_s42

  TASK-RELATIONSHIP ADVISORY
  ----------------------------------------
  Cross-task merge: adapters were evaluated on different tasks (qnli_dev
  vs rte_dev). Linear merges across task boundaries may degrade the
  weaker task's performance.

  2. BEHAVIORAL STATUS
  ----------------------------------------
  Adapter A eligibility: eligible
  Adapter B eligibility: eligible

  Confidence:      medium
                   Moderate spectral compatibility (score=0.313) — both adapters have verified behavioral quality

  3. ELIGIBILITY WARNING
  ----------------------------------------
  No eligibility concerns detected.

  4. RECOMMENDED ACTION
  ----------------------------------------
  Merge with audit-aware strategy. Consider norm-equalized as simpler alternative.
  Strategy: norm_equalized
```

The `TASK-RELATIONSHIP ADVISORY` section appears between Structural Result and Behavioral Status. It names both tasks and warns about weaker-task degradation.

---

## Example 3: Inventory-level interpretation note

In a 7-adapter inventory spanning SST-2, QNLI, and RTE:

> 21 pairs total. 16 carry the task-relationship advisory; 5 do not.
>
> The 5 advisory-free pairs are the same-task pairs: SST-2 x SST-2 (1 pair), QNLI x QNLI (3 pairs), RTE x RTE (1 pair). These are the safe merge candidates.
>
> The 16 advisory-bearing pairs span all cross-task combinations. Structural risk varies (4 high, 12 medium), but all share the same metadata warning: these adapters were trained on different tasks and linear merging may degrade the weaker task.
>
> The advisory partitions the 21-pair matrix into a clear structure: 5 same-task safe pairs and 16 cross-task caution pairs. This is the same compression that neighborhoods provide, but from an orthogonal signal.
